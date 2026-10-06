
import json
import os
import pickle
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from config import PROCESSED_PATH, PRED_PATH, TAB_PATH, DATA_PATH, RAW, N_JOBS, SEED, ALARM
from lib.alarms import smooth, alarms, conformal_threshold, min_threshold_for_budget, THRESH_GRID
from lib.features import extract
from lib.metrics import youden_threshold
from lib.zoo import CARES, make_rocket

DELTA = ALARM["delta"]


# ====================================================================================================== FoG
def fog_segments(t):
    """Split a person's window start times into contiguous recording segments."""
    cut = np.where(np.diff(t) > 2.0)[0] + 1
    return np.split(np.arange(len(t)), cut)


def fog_eval(t_end, score, eps, thr, k, refr, tol=2.0):
    """Alarm statistics for one person. eps: list of (start, end) freeze episodes (s)."""
    al = []
    for seg in fog_segments(t_end):
        al.append(alarms(t_end[seg], smooth(score[seg], k), thr, refr))
    al = np.concatenate(al) if al else np.zeros(0)
    hours = len(t_end) / 3600.0                           # 1-s hop -> one window per second of monitoring
    inside = np.zeros(len(al), bool)
    det, lat = [], []
    for s, e in eps:
        m = (al >= s - tol) & (al <= e + 4.0)
        inside |= m
        det.append(bool(m.any()))
        if m.any():
            lat.append(al[m].min() - s)
    fa = int((~inside).sum())
    return dict(fa=fa, hours=hours, fa_per_h=fa / hours, episodes=len(eps), detected=int(np.sum(det)),
                latency=lat)


def run_fog():
    d = np.load(os.path.join(PRED_PATH, "bench_fog.npz"))
    w = np.load(os.path.join(PROCESSED_PATH, "fog_windows.npz"))
    eps_all = {int(k): v for k, v in json.load(open(os.path.join(PROCESSED_PATH, "fog_episodes.json"))).items()}
    y, subj, t = d["y"], d["subject"], w["t"] + 4.0      # alarm time = end of the window
    model = "CARES (dual-view)"
    P = d[f"P::{model}"][:, 1]
    k, refr = ALARM["smooth_k"]["fog"], ALARM["refractory_s"]["fog"]
    budgets = [2.0, 4.0, 8.0, 16.0, 32.0]
    rows, lat_rows = [], []
    for s in np.unique(subj):
        te = subj == s
        tr_idx = np.where(~te)[0]
        inner = d[f"inner::{model}|{s}"]
        # per calibration person: theta_j for every budget
        th_j = {B: [] for B in budgets}
        pooled = []
        for j in np.unique(subj[tr_idx]):
            jj = subj[tr_idx] == j
            tj, sj = t[tr_idx][jj], inner[jj]
            ej = eps_all.get(int(j), [])
            fn = (lambda th, tj=tj, sj=sj, ej=ej: fog_eval(tj, sj, ej, th, k, refr)["fa_per_h"])
            rates = np.array([fn(th) for th in THRESH_GRID])
            pooled.append((rates, len(tj) / 3600.0))
            for B in budgets:
                ok = rates <= B + 1e-12
                adm = np.flip(np.logical_and.accumulate(np.flip(ok)))
                th_j[B].append(THRESH_GRID[np.where(adm)[0][0]] if adm.any() else 1.0)
        thr_youden = float(d[f"thr::{model}"][te][0])
        policies = {"fixed 0.5": {B: 0.5 for B in budgets}, "Youden (window-optimal)": {B: thr_youden for B in budgets}}
        # naive: one threshold meeting the budget on the POOLED calibration hours (no per-person guarantee)
        tot_h = sum(h for _, h in pooled)
        pooled_rate = sum(r * h for r, h in pooled) / tot_h
        naive = {}
        for B in budgets:
            ok = pooled_rate <= B + 1e-12
            adm = np.flip(np.logical_and.accumulate(np.flip(ok)))
            naive[B] = THRESH_GRID[np.where(adm)[0][0]] if adm.any() else 1.0
        policies["naive pooled calibration"] = naive
        policies["conformal (proposed)"] = {B: min(conformal_threshold(th_j[B], DELTA), 1.0) for B in budgets}
        for pol, thb in policies.items():
            for B in budgets:
                r = fog_eval(t[te], P[te], eps_all.get(int(s), []), thb[B], k, refr)
                rows.append(dict(policy=pol, budget_per_h=B, subject=s, threshold=thb[B], fa_per_h=r["fa_per_h"],
                                 within_budget=r["fa_per_h"] <= B, episodes=r["episodes"], detected=r["detected"],
                                 hours=r["hours"], median_latency_s=np.median(r["latency"]) if r["latency"] else np.nan))
                lat_rows += [dict(policy=pol, budget_per_h=B, subject=s, latency_s=v) for v in r["latency"]]
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(TAB_PATH, "S7_alarm_per_subject_fog.csv"), index=False)
    L = pd.DataFrame(lat_rows)
    summ = R.groupby(["policy", "budget_per_h"]).apply(lambda g: pd.Series(dict(
        coverage=g.within_budget.mean(), median_fa_per_h=g.fa_per_h.median(), max_fa_per_h=g.fa_per_h.max(),
        pooled_fa_per_h=(g.fa_per_h * g.hours).sum() / g.hours.sum(),
        episode_sensitivity=g.detected.sum() / max(g.episodes.sum(), 1),
        median_threshold=g.threshold.median()))).reset_index()
    lm = L.groupby(["policy", "budget_per_h"]).latency_s.median().rename("median_latency_s").reset_index()
    summ = summ.merge(lm, on=["policy", "budget_per_h"], how="left")
    summ.to_csv(os.path.join(TAB_PATH, "T6_alarm_budget_fog.csv"), index=False)
    print(summ.round(3).to_string(index=False))


# ====================================================================================================== falls
def stream_windows():
    from lib.falls_io import load
    d = load()
    sig, off = d["signal"], d["offsets"]
    fs, L, step = 25, 100, 10
    Xs, meta = [], []
    for i in range(len(d["subject"])):
        s = sig[off[i]:off[i + 1]][:, :12]               # waist (acc, gyr) + wrist (acc, gyr)
        if len(s) < L:
            s = np.pad(s, ((0, L - len(s)), (0, 0)), mode="edge")
        peak = int(np.argmax(np.linalg.norm(s[:, 0:3], axis=1)))
        fall = d["activity"][i] >= 900
        for a in range(0, len(s) - L + 1, step):
            Xs.append(s[a:a + L])
            inside = a + 25 <= peak < a + 75
            meta.append((d["subject"][i], d["activity"][i], d["repetition"][i], i, fall, (a + L) / fs, peak / fs,
                         int(fall and inside), int(not fall)))
    X = np.stack(Xs).astype(np.float32)
    M = pd.DataFrame(meta, columns=["subject", "activity", "rep", "trial", "fall", "t_end", "t_peak", "pos", "adl"])
    return X, M


def falls_fold(F, R, M, s):
    cache = os.path.join(PRED_PATH, "cache", f"stream_falls_{s}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    out = _falls_fold(F, R, M, s)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    pickle.dump(out, open(cache + ".tmp", "wb"))
    os.replace(cache + ".tmp", cache)
    print("fold done falls", s, flush=True)
    return out


def _falls_fold(F, R, M, s):
    te = (M.subject == s).to_numpy()
    trn = (~te) & ((M.pos == 1) | (M.adl == 1)).to_numpy()
    y = M.pos.to_numpy()
    mr = make_rocket("falls", R[trn][::4], seed=SEED)
    Mf = mr.transform(R)
    m = CARES(SEED).fit(F[trn], Mf[trn], y[trn], 2)
    p_te = m.predict_proba(F[te], Mf[te])[:, 1]
    # inner OOF scores on all training-person windows (used for calibration)
    tr_people = (~te)
    idx_tr = np.where(tr_people)[0]
    oof = np.full(len(idx_tr), np.nan)
    g = M.subject.to_numpy()[idx_tr]
    for a, b in GroupKFold(3).split(idx_tr, groups=g):
        fit_idx = idx_tr[a][((M.pos.to_numpy()[idx_tr[a]] == 1) | (M.adl.to_numpy()[idx_tr[a]] == 1))]
        mi = CARES(SEED).fit(F[fit_idx], Mf[fit_idx], y[fit_idx], 2)
        oof[b] = mi.predict_proba(F[idx_tr[b]], Mf[idx_tr[b]])[:, 1]
    return dict(subject=s, p_te=p_te, idx_te=np.where(te)[0], oof=oof, idx_tr=idx_tr)


def falls_trial_eval(M, p, thr, refr):
    """Per trial: alarms (with refractory) -> fall detected within +-2 s of impact, false alarms on ADL trials."""
    out = []
    for tri, g in M.assign(p=p).groupby("trial", sort=False):
        al = alarms(g.t_end.to_numpy(), g.p.to_numpy(), thr, refr)
        fall = bool(g.fall.iloc[0])
        pk = g.t_peak.iloc[0]
        det = bool(((al >= pk - 0.5) & (al <= pk + 4.5)).any()) if fall else False
        lat = (al[(al >= pk - 0.5) & (al <= pk + 4.5)].min() - pk) if det else np.nan
        dur = g.t_end.max() - g.t_end.min() + 4.0
        out.append(dict(trial=tri, subject=g.subject.iloc[0], activity=g.activity.iloc[0], fall=fall, detected=det,
                        latency_s=lat, false_alarms=0 if fall else len(al), adl_hours=0.0 if fall else dur / 3600))
    return pd.DataFrame(out)


def adl_fa_rates(Mj, pj, grid, refr):
    """False alarms per ADL hour of one person for every threshold of the grid (exact, with refractory period)."""
    adl = (Mj.adl == 1).to_numpy()
    tri = Mj.trial.to_numpy()[adl]; te = Mj.t_end.to_numpy()[adl]; pp = pj[adl]
    groups = np.split(np.arange(len(tri)), np.where(np.diff(tri) != 0)[0] + 1)
    hours = sum((te[g].max() - te[g].min() + 4.0) for g in groups) / 3600.0
    counts = np.zeros(len(grid))
    for g in groups:
        if pp[g].max() < grid[0]:
            continue
        for k, th in enumerate(grid):
            if pp[g].max() < th:
                break
            counts[k] += len(alarms(te[g], pp[g], th, refr))
    return counts / hours


def run_falls():
    t0 = time.time()
    X, M = stream_windows()
    print("streaming windows", X.shape, flush=True)
    sens = {"waist_acc": [0, 1, 2], "waist_gyr": [3, 4, 5], "wrist_acc": [6, 7, 8], "wrist_gyr": [9, 10, 11]}
    F, _ = extract(X, 25.0, sens)
    R = np.stack([np.linalg.norm(X[:, :, c:c + 3], axis=2) for c in (0, 3, 6, 9)], -1).astype(np.float32)
    del X
    print("features", F.shape, "%.1f min" % ((time.time() - t0) / 60), flush=True)
    subs = np.unique(M.subject)
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(falls_fold)(F, R, M, s) for s in subs)
    P = np.zeros(len(M))
    for r in res:
        P[r["idx_te"]] = r["p_te"]
    np.savez_compressed(os.path.join(PRED_PATH, "stream_falls.npz"), p=P, **{c: M[c].to_numpy() for c in M.columns})
    refr = ALARM["refractory_s"]["falls"]
    budgets = [0.0, 2.0, 6.0]                            # false alarms per hour of ADL
    rows, trials_all = [], []
    for r in res:
        s = r["subject"]
        Mt = M.iloc[r["idx_te"]]
        Mi = M.iloc[r["idx_tr"]]
        th_j = {B: [] for B in budgets}
        for j in np.unique(Mi.subject):
            jj = (Mi.subject == j).to_numpy()
            rates = adl_fa_rates(Mi[jj], r["oof"][jj], THRESH_GRID, refr)
            for B in budgets:
                ok = rates <= B + 1e-12
                adm = np.flip(np.logical_and.accumulate(np.flip(ok)))
                th_j[B].append(THRESH_GRID[np.where(adm)[0][0]] if adm.any() else 1.0)
        yy = Mi.pos.to_numpy(); keep = ((Mi.pos == 1) | (Mi.adl == 1)).to_numpy()
        thr_y = youden_threshold(yy[keep], r["oof"][keep])
        pol = {"fixed 0.5": {B: 0.5 for B in budgets}, "Youden (window-optimal)": {B: thr_y for B in budgets},
               "conformal (proposed)": {B: min(conformal_threshold(th_j[B], DELTA), 1.0) for B in budgets}}
        # personal ADL margin: first 5 minutes of the new person's ADL trials (ordered by protocol) are onboarding
        adl_tr = Mt[Mt.adl == 1].sort_values(["activity", "rep"]).trial.unique()
        cum, onboard = 0.0, []
        for tri in adl_tr:
            if cum >= 300:
                break
            onboard.append(tri)
            cum += (Mt.trial == tri).sum() * 0.4 + 3.6
        on_mask = Mt.trial.isin(onboard).to_numpy()
        pmax = r["p_te"][on_mask].max() if on_mask.any() else 0.0
        pol["conformal + personal ADL margin"] = {B: max(pol["conformal (proposed)"][B], min(pmax + 1e-3, 1.0)) for B in budgets}
        for name, thb in pol.items():
            for B in budgets:
                ev_mask = ~on_mask if name == "conformal + personal ADL margin" else np.ones(len(Mt), bool)
                # every policy is scored on the same trials (onboarding trials excluded for all)
                ev_mask = ~on_mask
                tr_eval = falls_trial_eval(Mt[ev_mask], r["p_te"][ev_mask], thb[B], refr)
                fa_h = tr_eval.false_alarms.sum() / tr_eval.adl_hours.sum()
                f = tr_eval[tr_eval.fall]
                rows.append(dict(policy=name, budget_per_h=B, subject=s, threshold=thb[B], fa_per_h=fa_h,
                                 within_budget=fa_h <= B + 1e-9, falls=len(f), detected=int(f.detected.sum()),
                                 adl_hours=tr_eval.adl_hours.sum(), adl_trials=int((~tr_eval.fall).sum()),
                                 adl_trials_with_alarm=int(((~tr_eval.fall) & (tr_eval.false_alarms > 0)).sum()),
                                 median_latency_s=f.latency_s.median()))
                if name == "conformal (proposed)" and B == 0.0:
                    trials_all.append(tr_eval.assign(policy=name))
    Rr = pd.DataFrame(rows)
    Rr.to_csv(os.path.join(TAB_PATH, "S8_alarm_per_subject_falls.csv"), index=False)
    summ = Rr.groupby(["policy", "budget_per_h"]).apply(lambda g: pd.Series(dict(
        coverage=g.within_budget.mean(), pooled_fa_per_h=(g.fa_per_h * g.adl_hours).sum() / g.adl_hours.sum(),
        median_fa_per_h=g.fa_per_h.median(), adl_trials_with_alarm_pct=100 * g.adl_trials_with_alarm.sum() / g.adl_trials.sum(),
        fall_sensitivity=g.detected.sum() / g.falls.sum(), median_latency_s=g.median_latency_s.median(),
        median_threshold=g.threshold.median()))).reset_index()
    summ.to_csv(os.path.join(TAB_PATH, "T6_alarm_budget_falls.csv"), index=False)
    TA = pd.concat(trials_all)
    TA.to_csv(os.path.join(TAB_PATH, "S9b_falls_trials_conformal.csv"), index=False)
    act = TA.groupby(["activity", "fall"]).agg(trials=("trial", "size"), detected=("detected", "sum"),
                                              trials_with_false_alarm=("false_alarms", lambda x: int((x > 0).sum()))).reset_index()
    act.to_csv(os.path.join(TAB_PATH, "S9_falls_by_activity.csv"), index=False)
    print(summ.round(3).to_string(index=False))
    print("elapsed %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    {"fog": run_fog, "falls": run_falls}[sys.argv[1]]()
