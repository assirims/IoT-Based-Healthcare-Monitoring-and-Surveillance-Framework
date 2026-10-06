
import os
import pickle
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per worker: folds run in parallel instead

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from config import PRED_PATH, TAB_PATH, N_JOBS, SEED
from lib.metrics import binary_metrics, multiclass_metrics, youden_threshold, cluster_bootstrap, wilcoxon_p, holm
from lib.zoo import load_service, make_models, make_rocket, fit_predict

INNER = 3


def run_fold(service, d, test_subject):
    """One LOSO fold. Results are cached on disk so that an interrupted run resumes where it stopped."""
    cache = os.path.join(PRED_PATH, "cache", f"bench_{service}_{test_subject}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    te = d["subject"] == test_subject
    tr = ~te
    K = d["K"]
    mr = make_rocket(service, d["R"][tr], seed=SEED)
    M = mr.transform(d["R"])
    out = dict(subject=test_subject, probs={}, thr={}, fit_s={}, infer_ms={}, size={}, inner={})
    # CARES = 0.5 * (HGB on descriptors + MiniRocket-LR); its two members are fitted once and re-used
    models = [m for m in make_models(service, d["names"], seed=SEED) if m.uses != "FM"]
    tr_idx = np.where(tr)[0]
    binary = K == 2
    for m in models:
        P, ft, it = fit_predict(m, d["F"][tr], M[tr], d["y"][tr], d["F"][te], M[te], K)
        out["probs"][m.name] = P.astype(np.float32)
        out["fit_s"][m.name] = ft
        out["infer_ms"][m.name] = it * 1000
        if test_subject == np.unique(d["subject"])[0]:
            out["size"][m.name] = m.size_bytes()
        if binary:
            # inner subject-grouped CV on the training people -> operating threshold + calibration scores
            oof = np.full(tr.sum(), np.nan)
            g = d["subject"][tr]
            for itr, iva in GroupKFold(n_splits=min(INNER, len(np.unique(g)))).split(tr_idx, groups=g):
                mm = [x for x in make_models(service, d["names"], seed=SEED) if x.name == m.name][0]
                a, b = tr_idx[itr], tr_idx[iva]
                Pi, _, _ = fit_predict(mm, d["F"][a], M[a], d["y"][a], d["F"][b], M[b], K)
                oof[iva] = Pi[:, 1]
            out["thr"][m.name] = youden_threshold(d["y"][tr], oof)
            out["inner"][m.name] = oof.astype(np.float32)
    c, a_, b_ = "CARES (dual-view)", "HGB", "MiniRocket-LR"
    out["probs"][c] = (0.5 * (out["probs"][a_] + out["probs"][b_])).astype(np.float32)
    out["fit_s"][c] = out["fit_s"][a_] + out["fit_s"][b_]
    out["infer_ms"][c] = out["infer_ms"][a_] + out["infer_ms"][b_]
    if out["size"]:
        out["size"][c] = out["size"][a_] + out["size"][b_]
    if binary:
        out["inner"][c] = (0.5 * (out["inner"][a_] + out["inner"][b_])).astype(np.float32)
        out["thr"][c] = youden_threshold(d["y"][tr], out["inner"][c])
    out["mr_features"] = M.shape[1]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    pickle.dump(out, open(cache + ".tmp", "wb"))
    os.replace(cache + ".tmp", cache)
    print("fold done", service, test_subject, flush=True)
    return out


def main(service):
    t0 = time.time()
    d = load_service(service)
    subs = np.unique(d["subject"])
    print(service, "windows", len(d["y"]), "subjects", len(subs), "features", d["F"].shape[1], flush=True)
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_fold)(service, d, s) for s in subs)
    names = list(res[0]["probs"].keys())
    K = d["K"]
    N = len(d["y"])
    probs = {m: np.zeros((N, K), np.float32) for m in names}
    thr = {m: np.full(N, np.nan, np.float32) for m in names}
    inner = {}
    for r in res:
        te = d["subject"] == r["subject"]
        for m in names:
            probs[m][te] = r["probs"][m]
            if r["thr"]:
                thr[m][te] = r["thr"][m]
                inner[f"{m}|{r['subject']}"] = r["inner"][m]
    np.savez_compressed(os.path.join(PRED_PATH, f"bench_{service}.npz"), y=d["y"], subject=d["subject"], t=d["t"],
                        models=np.array(names), **{f"P::{m}": probs[m] for m in names},
                        **{f"thr::{m}": thr[m] for m in names}, **{f"inner::{k}": v for k, v in inner.items()})
    # ---------------- metrics ----------------
    rows, per = [], []
    for m in names:
        if K == 2:
            p = probs[m][:, 1]
            yh_thr = thr[m]
            met = binary_metrics(d["y"], p, 0.5)
            # pooled operating-point metrics with each subject's own (training-derived) threshold
            yh = (p >= yh_thr).astype(int)
            tp = ((yh == 1) & (d["y"] == 1)).sum(); fn = ((yh == 0) & (d["y"] == 1)).sum()
            tn = ((yh == 0) & (d["y"] == 0)).sum(); fp = ((yh == 1) & (d["y"] == 0)).sum()
            met.update(Sens=tp / (tp + fn), Spec=tn / (tn + fp), PPV=tp / max(tp + fp, 1), F1=2 * tp / (2 * tp + fp + fn))
            met["BalAcc"] = (met["Sens"] + met["Spec"]) / 2
            from sklearn.metrics import roc_auc_score, average_precision_score
            lo, hi = cluster_bootstrap(d["subject"], lambda ii: roc_auc_score(d["y"][ii], p[ii]), n_boot=500, seed=1)
            met["AUROC_lo"], met["AUROC_hi"] = lo, hi
            lo, hi = cluster_bootstrap(d["subject"], lambda ii: average_precision_score(d["y"][ii], p[ii]), n_boot=500, seed=1)
            met["AUPRC_lo"], met["AUPRC_hi"] = lo, hi
            for s in subs:
                i = d["subject"] == s
                bm = binary_metrics(d["y"][i], p[i], float(yh_thr[i][0]))
                per.append(dict(model=m, subject=s, **bm))
        else:
            met = multiclass_metrics(d["y"], probs[m], d["classes"])
            from sklearn.metrics import f1_score
            lo, hi = cluster_bootstrap(d["subject"], lambda ii: f1_score(d["y"][ii], probs[m][ii].argmax(1), average="macro"), n_boot=500, seed=1)
            met["MacroF1_lo"], met["MacroF1_hi"] = lo, hi
            for s in subs:
                i = d["subject"] == s
                per.append(dict(model=m, subject=s, **multiclass_metrics(d["y"][i], probs[m][i], d["classes"])))
        met.update(model=m, fit_s_mean=np.mean([r["fit_s"][m] for r in res]),
                   infer_ms_per_window=np.mean([r["infer_ms"][m] for r in res]), size_kB=res[0]["size"].get(m, np.nan) / 1024)
        rows.append(met)
    T = pd.DataFrame(rows)
    P = pd.DataFrame(per)
    # paired Wilcoxon (subject level) of the proposed model vs every comparator
    key = "AUROC" if K == 2 else "MacroF1"
    ref = "CARES (dual-view)"
    pv = []
    for m in names:
        a = P[P.model == ref].set_index("subject")[key]
        b = P[P.model == m].set_index("subject")[key].reindex(a.index)
        pv.append(wilcoxon_p(a.values, b.values) if m != ref else np.nan)
    T["p_vs_CARES"] = pv
    T["p_holm"] = holm(pv)
    T["mr_features"] = res[0]["mr_features"]
    T.to_csv(os.path.join(TAB_PATH, f"T3_benchmark_{service}.csv"), index=False)
    P.to_csv(os.path.join(TAB_PATH, f"S2_per_subject_{service}.csv"), index=False)
    cols = [c for c in ["model", "AUROC", "AUROC_lo", "AUROC_hi", "AUPRC", "Sens", "Spec", "F1", "MacroF1", "MacroF1_lo",
                        "MacroF1_hi", "BalAcc", "Kappa", "p_holm", "fit_s_mean", "infer_ms_per_window", "size_kB"] if c in T]
    print(T[cols].round(4).to_string(index=False))
    print("elapsed %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main(sys.argv[1])
