
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import decimate
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score

from config import PROCESSED_PATH, TAB_PATH, PRED_PATH, N_JOBS, SEED, HGB
from lib.features import extract, STAT_NAMES

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
FE = import_module("07_extract_edge_features")

FAMILY = {"time": ["mean", "std", "min", "max", "p10", "p50", "p90", "iqr", "skew", "kurt", "rms", "jerk", "zcr", "sma"],
          "spectral": ["dom_freq", "spec_entropy", "bp_0.5_3", "bp_3_8", "bp_8_12", "freeze_index", "total_power"],
          "gait regularity": ["acf_peak", "acf_lag"], "posture/coordination": ["corr_xy", "corr_xz", "corr_yz"]}


def family_of(name):
    last = name.split(":")[-1]
    for f, lst in FAMILY.items():
        if last in lst:
            return f
    return "time"


def site_of(name):
    return name.split(":")[0].split("_")[0]


def hgb(K=2):
    # lighter boosting than the benchmark (150 / 80 iterations) - the analyses compare settings with the SAME model
    if K == 2:
        return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, max_iter=150, learning_rate=0.1,
                                              max_leaf_nodes=31, l2_regularization=1.0)
    return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, max_iter=80, learning_rate=0.12,
                                          max_leaf_nodes=31, l2_regularization=1.0)


def _one(F, y, subj, K, s, test_F, mask_cols):
    te = subj == s
    m = hgb(K).fit(F[~te], y[~te])
    Ft = (F if test_F is None else test_F)[te].copy()
    if mask_cols is not None:
        Ft[:, mask_cols] = np.nan
    return np.where(te)[0], m.predict_proba(Ft), m.classes_


def loso_scores(F, y, subj, K, key, test_F=None, mask_cols=None):
    """LOSO predictions of the descriptor-branch model (cached). test_F: perturbed test inputs (same columns)."""
    cache = os.path.join(PRED_PATH, "cache", f"rob_{key}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    P = np.zeros((len(y), K))
    res = Parallel(n_jobs=N_JOBS)(delayed(_one)(F, y, subj, K, s, test_F, mask_cols) for s in np.unique(subj))
    for idx, p, cls in res:
        P[idx[:, None], cls[None, :]] = p
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, P)
    return P


def score(y, P, K):
    if K == 2:
        return roc_auc_score(y, P[:, 1])
    return f1_score(y, P.argmax(1), average="macro")


def load(service):
    f = np.load(os.path.join(PROCESSED_PATH, f"{service}_features.npz"))
    w = np.load(os.path.join(PROCESSED_PATH, f"{service}_windows.npz"))
    K = len(f["classes"]) if "classes" in f.files else 2
    return f, w, K


def main(service):
    t0 = time.time()
    f, w, K = load(service)
    names = f["names"]
    blocks = np.array([n.split(":")[0] for n in names])
    y, subj = f["y"].astype(int), f["subject"].astype(int)
    kit = FE.KIT[service]
    F = f["F"]
    rows = []
    # ---------------- (a) sensor placement ----------------
    if service == "falls":
        sites = ["waist", "wrist", "chest", "thigh", "ankle", "head"]
        configs = {f"{s} (acc+gyr)": [f"{s}_acc", f"{s}_gyr"] for s in sites}
        configs.update({f"{s} (acc only)": [f"{s}_acc"] for s in ["waist", "wrist"]})
        configs["waist+wrist (CARES kit)"] = kit
        configs["waist+wrist (acc only)"] = ["waist_acc", "wrist_acc"]
        configs["all six sites"] = list(np.unique(blocks))
    elif service == "fog":
        configs = {"ankle": ["ankle"], "thigh": ["thigh"], "trunk": ["trunk"], "ankle+thigh": ["ankle", "thigh"],
                   "ankle+trunk": ["ankle", "trunk"], "all three (CARES kit)": kit}
    else:
        configs = {"lower back": ["back"], "thigh": ["thigh"], "back+thigh (CARES kit)": kit}

    res = []
    for nm, bl in configs.items():
        cols = np.isin(blocks, bl)
        P = loso_scores(F[:, cols], y, subj, K, key=f"{service}_place_{nm}".replace(" ", "_").replace("/", "-"))
        res.append((nm, score(y, P, K), P))
        print("  ", nm, round(res[-1][1], 4), flush=True)
    full_P = None
    for nm, sc, P in res:
        rows.append(dict(analysis="(a) sensor placement", setting=nm, score=sc))
        if "CARES kit" in nm:
            full_P = P
    print("placement done %.1f min" % ((time.time() - t0) / 60), flush=True)
    # ---------------- (b) graceful degradation ----------------
    kcols = np.isin(blocks, kit)
    Fk, bk = F[:, kcols], blocks[kcols]
    sensors_kit = sorted(set(s.split("_")[0] for s in kit))
    for lost in (sensors_kit if K == 2 else []):          # multi-class activity: covered by (a)
        lost_blocks = [b for b in kit if b.split("_")[0] == lost]
        mcols = np.where(np.isin(bk, lost_blocks))[0]
        P1 = loso_scores(Fk, y, subj, K, key=f"{service}_drop_missing_{lost}", mask_cols=mcols)
        keep = ~np.isin(bk, lost_blocks)
        P2 = loso_scores(Fk[:, keep], y, subj, K, key=f"{service}_drop_bank_{lost}")
        rows.append(dict(analysis="(b) sensor dropout", setting=f"{lost} lost: same model, missing inputs", score=score(y, P1, K)))
        rows.append(dict(analysis="(b) sensor dropout", setting=f"{lost} lost: CARES model bank", score=score(y, P2, K)))
    print("degradation done %.1f min" % ((time.time() - t0) / 60), flush=True)
    # ---------------- (c) sampling rate ----------------
    fs = float(w["fs"])
    X = w["X"]
    sens_all = FE.sensors_for(service)
    sens = {k: v for k, v in sens_all.items() if k in kit}
    for q in ([2, 4] if fs >= 50 else [2]):
        Xd = decimate(X, q, axis=1, zero_phase=True).astype(np.float32)
        Fd, nd = extract(Xd, fs / q, sens)
        P = loso_scores(Fd, y, subj, K, key=f"{service}_rate_{q}")
        rows.append(dict(analysis="(c) sampling rate", setting=f"{fs / q:g} Hz (from {fs:g} Hz)", score=score(y, P, K)))
    rows.append(dict(analysis="(c) sampling rate", setting=f"{fs:g} Hz (native)", score=score(y, full_P, K)))
    print("sampling done %.1f min" % ((time.time() - t0) / 60), flush=True)
    # ---------------- (d) test-time noise ----------------
    rng = np.random.default_rng(SEED)
    ch_sd = X.reshape(-1, X.shape[2]).std(0)               # noise relative to every channel's signal SD
    for sig in [0.1, 0.25, 0.5]:
        Xn = X + (rng.standard_normal(size=X.shape) * (sig * ch_sd)[None, None, :]).astype(np.float32)
        Fn, _ = extract(Xn, fs, sens)
        P = loso_scores(Fk, y, subj, K, key=f"{service}_noise_{sig}", test_F=Fn)
        rows.append(dict(analysis="(d) sensor noise (test only)", setting=f"noise SD = {int(sig * 100)}% of signal SD", score=score(y, P, K)))
    print("noise done %.1f min" % ((time.time() - t0) / 60), flush=True)
    R = pd.DataFrame(rows)
    R["metric"] = "AUROC" if K == 2 else "macro-F1"
    R.to_csv(os.path.join(TAB_PATH, f"T8_robustness_{service}.csv"), index=False)
    print(R.round(4).to_string(index=False))
    # ---------------- (e) alert explanations (binary services) ----------------
    if K == 2:
        nk = names[kcols]
        groups = {}
        for i, n in enumerate(nk):
            groups.setdefault((site_of(n), family_of(n)), []).append(i)
        expl, faith = [], []
        rng = np.random.default_rng(SEED)
        for s in np.unique(subj):
            te = subj == s
            m = hgb(K).fit(Fk[~te], y[~te])
            base = Fk[~te].mean(0)
            Ft = Fk[te]
            p = m.predict_proba(Ft)[:, 1]
            pos = np.where(p >= 0.5)[0]
            if len(pos) == 0:
                continue
            pos = pos[:: max(1, len(pos) // 150)]
            Fp = Ft[pos]
            contrib = {}
            for g, cols in groups.items():
                Z = Fp.copy(); Z[:, cols] = base[cols]
                contrib[g] = p[pos] - m.predict_proba(Z)[:, 1]
            keys = list(contrib)
            Cm = np.stack([contrib[k] for k in keys], 1)
            top = Cm.argmax(1)
            for i, k in enumerate(top):
                expl.append(dict(subject=s, site=keys[k][0], family=keys[k][1], true_event=int(y[te][pos[i]]),
                                 top_contribution=Cm[i, k]))
            # faithfulness: deleting the top group vs. a random group
            for i in range(len(pos)):
                Z1 = Fp[i:i + 1].copy(); Z1[:, groups[keys[top[i]]]] = base[groups[keys[top[i]]]]
                rk = keys[rng.integers(len(keys))]
                Z2 = Fp[i:i + 1].copy(); Z2[:, groups[rk]] = base[groups[rk]]
                faith.append(dict(drop_top=p[pos[i]] - m.predict_proba(Z1)[0, 1], drop_random=p[pos[i]] - m.predict_proba(Z2)[0, 1]))
        E = pd.DataFrame(expl)
        Fa = pd.DataFrame(faith)
        E.groupby(["site", "family"]).size().rename("alerts").reset_index().sort_values("alerts", ascending=False).to_csv(
            os.path.join(TAB_PATH, f"T9_alert_explanations_{service}.csv"), index=False)
        from scipy import stats
        fs_ = pd.DataFrame([dict(n_alerts=len(Fa), mean_drop_top=Fa.drop_top.mean(), mean_drop_random=Fa.drop_random.mean(),
                                 wilcoxon_p=stats.wilcoxon(Fa.drop_top, Fa.drop_random).pvalue)])
        fs_.to_csv(os.path.join(TAB_PATH, f"T9b_explanation_faithfulness_{service}.csv"), index=False)
        print(E.groupby(["site", "family"]).size().sort_values(ascending=False).head(8))
        print(fs_.round(4).to_string(index=False))
    print("elapsed %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main(sys.argv[1])
