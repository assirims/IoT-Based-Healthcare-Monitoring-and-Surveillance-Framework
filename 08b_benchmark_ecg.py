
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_PATH, PRED_PATH, TAB_PATH, SEED, SEEDS, FL, HGB
from lib.fed import fedavg, finetune
from lib.minirocket import MiniRocketLite
from lib.nn import MLP, train_local

CLASSES = ["N", "S", "V", "F", "Q"]
K = 5
ONBOARD_S = 300.0


def aami_metrics(y, yh):
    out = dict(n=len(y), Acc=float((y == yh).mean()))
    f1s = []
    for c, nm in enumerate(CLASSES[:4]):
        tp = ((yh == c) & (y == c)).sum(); fn = ((yh != c) & (y == c)).sum(); fp = ((yh == c) & (y != c)).sum()
        se = tp / (tp + fn) if tp + fn else np.nan
        pp = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
        out[f"Se_{nm}"], out[f"PPV_{nm}"], out[f"F1_{nm}"] = se, pp, f1
        f1s.append(f1)
    out["MacroF1_NSVF"] = np.nanmean(f1s)
    out["MacroF1_NSV"] = np.nanmean(f1s[:3])
    return out


def undersample(y, seed, keep_n=0.2):
    rng = np.random.default_rng(seed)
    idx = np.where((y != 0) | (rng.random(len(y)) < keep_n))[0]
    return idx


class ECGModels:
    """Feature views: F = RR + morphology descriptors (56), M = MiniRocket-lite of the 180-Hz beat segment."""

    def __init__(self, seed):
        self.seed = seed

    def fit_all(self, F, M, y, full=True):
        self.full = full
        self.sc = StandardScaler().fit(F)
        Z = self.sc.transform(F)
        self.m = {}
        self.m["LR"] = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000).fit(Z, y)
        if full:
            self.m["RF"] = RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=2,
                                                  random_state=self.seed).fit(F, y)
        idx = undersample(y, self.seed)
        self.m["HGB"] = HistGradientBoostingClassifier(early_stopping=False, random_state=self.seed, **HGB).fit(F[idx], y[idx])
        self.scm = StandardScaler().fit(M)
        self.m["MiniRocket-LR"] = LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000).fit(self.scm.transform(M), y)
        if full:
            net = MLP(F.shape[1], (128, 64), K, seed=self.seed, dropout=0.1)
            opt = None
            for ep in range(15):
                opt = train_local(net, np.clip(Z, -8, 8), y, K, epochs=1, lr=1e-3, batch=256, seed=self.seed + ep, opt=opt)
            self.net = net
        return self

    def proba(self, F, M):
        Z = self.sc.transform(F)
        P = {}
        for nm in ["LR", "RF", "HGB"]:
            if nm in self.m:
                p = self.m[nm].predict_proba(Z if nm == "LR" else F)
                P[nm] = expand(p, self.m[nm].classes_)
        P["MiniRocket-LR"] = expand(self.m["MiniRocket-LR"].predict_proba(self.scm.transform(M)), self.m["MiniRocket-LR"].classes_)
        if self.full:
            P["MLP"] = self.net.predict_proba(np.clip(Z, -8, 8))
        P["CARES (dual-view)"] = 0.5 * (P["HGB"] + P["MiniRocket-LR"])
        return P


def expand(p, cls):
    P = np.zeros((len(p), K))
    P[:, cls] = p
    return P


def main():
    t0 = time.time()
    d = np.load(os.path.join(PROCESSED_PATH, "ecg_beats.npz"))
    F = np.concatenate([d["rr"], d["morph"], d["extra"]], 1).astype(np.float32)
    seg = d["seg"].astype(np.float32)
    y, rec, tt = d["label"].astype(int), d["record"].astype(int), d["t"]
    ds1, ds2 = np.isin(rec, d["ds1"]), np.isin(rec, d["ds2"])
    mr = MiniRocketLite(dilations=(1, 2, 4, 8), n_bias=1, seed=SEED).fit(seg[ds1][:, :, None])
    t1 = time.time()
    M = mr.transform(seg[:, :, None])
    print("MiniRocket features", M.shape, "%.0fs" % (time.time() - t1), flush=True)
    late = ds2 & (tt >= ONBOARD_S)         # evaluation beats common to every scenario (after the onboarding period)
    rows = []
    # ---------------- A. global models ----------------
    G = ECGModels(SEED).fit_all(F[ds1], M[ds1], y[ds1])
    PA = G.proba(F, M)
    for nm, P in PA.items():
        for part, msk in [("DS2 (all beats)", ds2), ("DS2 after 5 min", late)]:
            rows.append(dict(scenario="A. global (inter-patient)", model=nm, eval=part, **aami_metrics(y[msk], P[msk].argmax(1))))
    print("A done %.1f min" % ((time.time() - t0) / 60), flush=True)
    # ---------------- B. patient-adaptive (first 5 min labelled) ----------------
    PB = {nm: np.zeros((len(y), K)) for nm in ["LR", "HGB", "MiniRocket-LR", "CARES (dual-view)", "MLP"]}
    scz = G.sc
    for r in np.unique(rec[ds2]):
        loc = (rec == r) & (tt < ONBOARD_S)
        ev = (rec == r) & (tt >= ONBOARD_S)
        # trees / linear models: retrain on DS1 + the person's onboarding beats (up-weighted by replication x5)
        Fa = np.concatenate([F[ds1]] + [F[loc]] * 5); Ma = np.concatenate([M[ds1]] + [M[loc]] * 5)
        ya = np.concatenate([y[ds1]] + [y[loc]] * 5)
        Pm = ECGModels(SEED).fit_all(Fa, Ma, ya, full=False).proba(F[ev], M[ev])
        for nm in Pm:
            PB[nm][ev] = Pm[nm]
        # the neural model is personalised on-device by fine-tuning its output layer only
        mft = finetune(G.net, np.clip(scz.transform(F[loc]), -8, 8), y[loc], K, epochs=20, lr=1e-3, last_only=True, seed=SEED)
        PB["MLP"][ev] = mft.predict_proba(np.clip(scz.transform(F[ev]), -8, 8))
        print("  record", r, "%.1f min" % ((time.time() - t0) / 60), flush=True)
    for nm, P in PB.items():
        rows.append(dict(scenario="B. patient-adaptive (5-min onboarding)", model=nm, eval="DS2 after 5 min",
                         **aami_metrics(y[late], P[late].argmax(1))))
    # ---------------- C. federated (DS1 people = clients) ----------------
    sc = StandardScaler().fit(F[ds1])                       # in deployment: federated mean/variance (sums only)
    Z = np.clip(sc.transform(F), -8, 8)
    clients = [(Z[(rec == r)], y[rec == r]) for r in np.unique(rec[ds1])]
    fl_rows = []
    for mu, nmf in [(0.0, "FedAvg MLP"), (FL["mu"], "FedProx MLP")]:
        Ps, Pps = [], []
        for sd in SEEDS:
            g, hist = fedavg(clients, Z.shape[1], K, rounds=FL["rounds"], local_epochs=1, lr=FL["lr"], batch=FL["batch"],
                             hidden=FL["hidden"], mu=mu, seed=sd)
            Ps.append(g.predict_proba(Z))
            Pp = np.zeros((len(y), K))
            for r in np.unique(rec[ds2]):
                loc = (rec == r) & (tt < ONBOARD_S); ev = (rec == r) & (tt >= ONBOARD_S)
                m = finetune(g, Z[loc], y[loc], K, epochs=20, lr=1e-3, last_only=True, seed=sd)
                Pp[ev] = m.predict_proba(Z[ev])
            Pps.append(Pp)
            fl_rows.append(dict(model=nmf, seed=sd, **aami_metrics(y[late], Ps[-1][late].argmax(1))))
        P = np.mean(Ps, 0); Pp = np.mean(Pps, 0)
        PA[nmf] = P; PB[nmf + " + on-device fine-tuning"] = Pp
        rows.append(dict(scenario="C. federated (22 DS1 clients)", model=nmf, eval="DS2 after 5 min", **aami_metrics(y[late], P[late].argmax(1))))
        rows.append(dict(scenario="C. federated (22 DS1 clients)", model=nmf + " + on-device fine-tuning", eval="DS2 after 5 min",
                         **aami_metrics(y[late], Pp[late].argmax(1))))
    print("C done %.1f min" % ((time.time() - t0) / 60), flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(TAB_PATH, "T4_ecg_arrhythmia.csv"), index=False)
    pd.DataFrame(fl_rows).to_csv(os.path.join(TAB_PATH, "S3_ecg_federated_seeds.csv"), index=False)
    # ---------------- D. ectopy burden (VEB per hour) ----------------
    br = []
    for nm, P in [("CARES (dual-view), global", PA["CARES (dual-view)"]), ("CARES (dual-view), patient-adaptive", PB["CARES (dual-view)"])]:
        for r in np.unique(rec[ds2]):
            ev = (rec == r) & (tt >= ONBOARD_S)
            hours = (tt[ev].max() - ONBOARD_S) / 3600
            br.append(dict(model=nm, record=r, ref_veb_per_h=(y[ev] == 2).sum() / hours,
                           pred_veb_per_h=(P[ev].argmax(1) == 2).sum() / hours,
                           ref_sveb_per_h=(y[ev] == 1).sum() / hours, pred_sveb_per_h=(P[ev].argmax(1) == 1).sum() / hours))
    B = pd.DataFrame(br)
    summ = []
    for nm, g in B.groupby("model"):
        for k in ["veb", "sveb"]:
            rho = stats.spearmanr(g[f"ref_{k}_per_h"], g[f"pred_{k}_per_h"]).correlation
            summ.append(dict(model=nm, beat=k.upper(), spearman_rho=rho,
                             median_abs_error_per_h=np.median(np.abs(g[f"ref_{k}_per_h"] - g[f"pred_{k}_per_h"]))))
    B.to_csv(os.path.join(TAB_PATH, "S4_ecg_burden_per_record.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(TAB_PATH, "T4b_ecg_burden.csv"), index=False)
    np.savez_compressed(os.path.join(PRED_PATH, "bench_ecg.npz"), y=y, record=rec, t=tt,
                        **{f"PA::{k}": v.astype(np.float32) for k, v in PA.items()},
                        **{f"PB::{k}": v.astype(np.float32) for k, v in PB.items()})
    cols = ["scenario", "model", "eval", "Acc", "Se_S", "PPV_S", "Se_V", "PPV_V", "MacroF1_NSV"]
    print(T[cols].round(3).to_string(index=False))
    print(pd.DataFrame(summ).round(3).to_string(index=False))
    print("elapsed %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
