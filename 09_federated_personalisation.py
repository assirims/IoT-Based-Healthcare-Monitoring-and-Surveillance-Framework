
import os
import pickle
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from config import PRED_PATH, TAB_PATH, N_JOBS, SEEDS, FL, ONBOARD_MIN
from lib.fed import fedavg, finetune
from lib.metrics import wilcoxon_p, holm
from lib.nn import MLP, train_local
from lib.zoo import load_service


def onboarding_order(service, d, idx):
    """Indices of a person's windows in the order an onboarding session would record them."""
    if service == "falls":
        adl = idx[d["y"][idx] == 0]
        return adl[np.argsort(d["activity"][adl] * 10 + 0, kind="stable")]
    return idx[np.argsort(d["t"][idx], kind="stable")]


def window_minutes(service, d):
    if service == "fog":
        return 1.0 / 60.0          # 1-s hop between windows
    if service == "har70":
        return 2.5 / 60.0
    return None


def score(K, y, P):
    if K == 2:
        if len(np.unique(y)) < 2:
            return np.nan
        return roc_auc_score(y, P[:, 1])
    labs = np.unique(y)
    return f1_score(y, P.argmax(1), labels=labs, average="macro", zero_division=0)


def run_fold(service, d, s):
    cache = os.path.join(PRED_PATH, "cache", f"fl_{service}_{s}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    out = _run_fold(service, d, s)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    pickle.dump(out, open(cache + ".tmp", "wb"))
    os.replace(cache + ".tmp", cache)
    print("fold done", service, s, flush=True)
    return out


def _run_fold(service, d, s):
    K = d["K"]
    te = np.where(d["subject"] == s)[0]
    tr = np.where(d["subject"] != s)[0]
    order = onboarding_order(service, d, te)
    wpm = window_minutes(service, d)
    n_max = int(round(max(ONBOARD_MIN) / wpm)) if wpm else 0
    on_all = order[:n_max]
    ev = np.setdiff1d(te, on_all) if service != "falls" else te
    # federated standardisation: pooled mean / variance from per-client sums (identical to pooled StandardScaler)
    mu = d["F"][tr].mean(0); sd = d["F"][tr].std(0) + 1e-6
    Z = np.clip((d["F"] - mu) / sd, -8, 8)
    clients = [(Z[d["subject"] == c], d["y"][d["subject"] == c]) for c in np.unique(d["subject"][tr])]
    out = dict(subject=s, ev=ev, P={}, curves={})
    for sd_ in SEEDS:
        # central reference
        net = MLP(Z.shape[1], FL["hidden"], K, seed=sd_, dropout=FL["dropout"])
        opt = None
        for ep in range(FL["rounds"] // 3):
            opt = train_local(net, Z[tr], d["y"][tr], K, epochs=1, lr=1e-3, batch=FL["batch"], seed=sd_ + ep, opt=opt)
        out["P"].setdefault("Central MLP", []).append(net.predict_proba(Z[ev]))
        for nm, mu_, frac in [("FedAvg MLP", 0.0, 1.0), ("FedProx MLP", FL["mu"], 1.0), ("FedAvg MLP (50% gateways online)", 0.0, 0.5)]:
            g, hist = fedavg(clients, Z.shape[1], K, rounds=FL["rounds"], local_epochs=FL["local_epochs"], lr=FL["lr"],
                             batch=FL["batch"], hidden=FL["hidden"], mu=mu_, dropout=FL["dropout"], seed=sd_, frac=frac,
                             val=(lambda m: score(K, d["y"][ev], m.predict_proba(Z[ev]))) if (nm == "FedAvg MLP" and sd_ == SEEDS[0]) else None,
                             eval_every=2)
            out["P"].setdefault(nm, []).append(g.predict_proba(Z[ev]))
            if hist:
                out["curves"][nm] = hist
            if nm == "FedAvg MLP" and service != "falls":
                for m in ONBOARD_MIN[1:]:
                    on = order[:int(round(m / wpm))]
                    ft = finetune(g, Z[on], d["y"][on], K, epochs=15, lr=1e-3, last_only=False, seed=sd_)
                    out["P"].setdefault(f"FedAvg + FT ({m} min)", []).append(ft.predict_proba(Z[ev]))
                    # local-only model: trained from scratch on the onboarding minutes only
                    loc = MLP(Z.shape[1], FL["hidden"], K, seed=sd_, dropout=FL["dropout"])
                    lopt = None
                    for ep in range(30):
                        lopt = train_local(loc, Z[on], d["y"][on], K, epochs=1, lr=1e-3, batch=64, seed=sd_ + ep, opt=lopt)
                    out["P"].setdefault(f"Local only ({m} min)", []).append(loc.predict_proba(Z[ev]))
    out["P"] = {k: np.mean(v, 0).astype(np.float32) for k, v in out["P"].items()}
    return out


def cares_personal(service, d, bench, ev_by_subject):
    """Personal Platt recalibration of the CARES probabilities on the onboarding minutes (binary services)."""
    P0 = bench["P::CARES (dual-view)"]
    res = {}
    wpm = window_minutes(service, d)
    for s, ev in ev_by_subject.items():
        te = np.where(d["subject"] == s)[0]
        order = onboarding_order(service, d, te)
        res.setdefault("CARES (dual-view)", {})[s] = P0[ev]
        if d["K"] != 2 or service == "falls":
            continue
        for m in ONBOARD_MIN[1:]:
            on = order[:int(round(m / wpm))]
            yo = d["y"][on]
            if len(np.unique(yo)) < 2:
                res.setdefault(f"CARES + PC ({m} min)", {})[s] = P0[ev]      # nothing to learn: keep global model
                continue
            lg = np.log(np.clip(P0[on, 1], 1e-6, 1 - 1e-6) / np.clip(1 - P0[on, 1], 1e-6, 1))
            cal = LogisticRegression(C=1.0).fit(lg[:, None], yo)
            le = np.log(np.clip(P0[ev, 1], 1e-6, 1 - 1e-6) / np.clip(1 - P0[ev, 1], 1e-6, 1))
            p = cal.predict_proba(le[:, None])
            res.setdefault(f"CARES + PC ({m} min)", {})[s] = p
    return res


def main(service):
    t0 = time.time()
    d = load_service(service)
    subs = np.unique(d["subject"])
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_fold)(service, d, s) for s in subs)
    bench = np.load(os.path.join(PRED_PATH, f"bench_{service}.npz"))
    ev_by = {r["subject"]: r["ev"] for r in res}
    cp = cares_personal(service, d, bench, ev_by)
    K = d["K"]
    methods = list(res[0]["P"].keys()) + list(cp.keys())
    rows, per = [], []
    pooled = {m: ([], []) for m in methods}
    for r in res:
        s, ev = r["subject"], r["ev"]
        for m in methods:
            P = r["P"][m] if m in r["P"] else cp[m][s]
            per.append(dict(method=m, subject=s, score=score(K, d["y"][ev], P), n_eval=len(ev)))
            pooled[m][0].append(d["y"][ev]); pooled[m][1].append(P)
    for m in methods:
        y = np.concatenate(pooled[m][0]); P = np.concatenate(pooled[m][1])
        if K == 2:
            rows.append(dict(method=m, AUROC=roc_auc_score(y, P[:, 1]), AUPRC=average_precision_score(y, P[:, 1])))
        else:
            rows.append(dict(method=m, MacroF1=f1_score(y, P.argmax(1), average="macro"), Acc=float((P.argmax(1) == y).mean())))
    T = pd.DataFrame(rows)
    Pp = pd.DataFrame(per)
    med = Pp.groupby("method").score.agg(["median", "mean"]).rename(columns={"median": "subject_median", "mean": "subject_mean"})
    T = T.merge(med, left_on="method", right_index=True)
    ref = "FedAvg MLP"
    a = Pp[Pp.method == ref].set_index("subject").score
    T["p_vs_FedAvg"] = [wilcoxon_p(a.values, Pp[Pp.method == m].set_index("subject").score.reindex(a.index).values) if m != ref else np.nan
                        for m in T.method]
    T["p_holm"] = holm(T["p_vs_FedAvg"].values)
    T.to_csv(os.path.join(TAB_PATH, f"T5_federated_personalised_{service}.csv"), index=False)
    Pp.to_csv(os.path.join(TAB_PATH, f"S5_federated_per_subject_{service}.csv"), index=False)
    curves = [dict(subject=r["subject"], round=rd, score=sc) for r in res for rd, sc in r["curves"].get("FedAvg MLP", [])]
    pd.DataFrame(curves).to_csv(os.path.join(TAB_PATH, f"S6_fedavg_convergence_{service}.csv"), index=False)
    print(T.round(4).to_string(index=False))
    print("elapsed %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main(sys.argv[1])
