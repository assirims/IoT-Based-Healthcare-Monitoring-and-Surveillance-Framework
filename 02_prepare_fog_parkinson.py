
import io
import json
import os
import zipfile

import numpy as np

from config import DATA_PATH, PROCESSED_PATH, RAW, WIN, TAB_PATH
from lib.windows import sliding_windows

CHANNELS = ["ankle_fwd", "ankle_vert", "ankle_lat", "thigh_fwd", "thigh_vert", "thigh_lat",
            "trunk_fwd", "trunk_vert", "trunk_lat"]
SENSORS = {"ankle": [0, 1, 2], "thigh": [3, 4, 5], "trunk": [6, 7, 8]}


def episodes(mask, t):
    """Start/end times of consecutive True runs."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return [(float(t[a]), float(t[b])) for a, b in zip(s, e)]


def main():
    cfg = WIN["fog"]
    z = zipfile.ZipFile(os.path.join(DATA_PATH, RAW["fog"]))
    names = sorted(n for n in z.namelist() if "/dataset/" in n and n.endswith(".txt"))
    Xs, ys, subj, run, ts, pur, cov = [], [], [], [], [], [], []
    eps, summary = {}, []
    t_offset = 0.0
    for n in names:
        base = os.path.basename(n)[:-4]                       # S01R01
        sid, rid = int(base[1:3]), int(base[4:6])
        a = np.loadtxt(io.BytesIO(z.read(n)))
        t = a[:, 0] / 1000.0
        sig = a[:, 1:10] / 1000.0                            # mg -> g
        lab = a[:, 10].astype(int)
        # windows are cut separately inside every contiguous protocol segment (label != 0)
        inprot = lab > 0
        segs = episodes(inprot, np.arange(len(lab)))
        for s0, s1 in segs:
            s0, s1 = int(s0), int(s1) + 1
            X, Y, st = sliding_windows(sig[s0:s1], lab[s0:s1], cfg["fs"], cfg["win_s"], cfg["step_s"], t0=t[s0])
            if len(X) == 0:
                continue
            frac = (Y == 2).mean(1)
            Xs.append(X); ys.append((frac >= 0.5).astype(np.int8))
            pur.append(np.maximum(frac, 1 - frac)); cov.append(np.ones(len(X)))
            subj.append(np.full(len(X), sid, np.int16)); run.append(np.full(len(X), rid, np.int16))
            ts.append(st + t_offset)
        fz = episodes(lab == 2, t + t_offset)
        eps.setdefault(sid, []).extend(fz)
        summary.append(dict(subject=sid, run=rid, minutes_protocol=round(inprot.sum() / cfg["fs"] / 60, 2),
                            minutes_freeze=round((lab == 2).sum() / cfg["fs"] / 60, 2), n_freeze_episodes=len(fz)))
        t_offset += t[-1] + 3600.0                            # keep runs apart on a common time axis
    X = np.concatenate(Xs); y = np.concatenate(ys)
    out = dict(X=X, y=y, subject=np.concatenate(subj), run=np.concatenate(run), t=np.concatenate(ts),
               purity=np.concatenate(pur).astype(np.float32), channels=np.array(CHANNELS), fs=cfg["fs"])
    np.savez_compressed(os.path.join(PROCESSED_PATH, "fog_windows.npz"), **out)
    json.dump({str(k): v for k, v in eps.items()}, open(os.path.join(PROCESSED_PATH, "fog_episodes.json"), "w"))
    import pandas as pd
    s = pd.DataFrame(summary)
    g = s.groupby("subject").agg(runs=("run", "count"), minutes_protocol=("minutes_protocol", "sum"),
                                 minutes_freeze=("minutes_freeze", "sum"), n_freeze_episodes=("n_freeze_episodes", "sum")).reset_index()
    w = pd.DataFrame(dict(subject=out["subject"], y=y)).groupby("subject").agg(windows=("y", "size"), freeze_windows=("y", "sum")).reset_index()
    g = g.merge(w, on="subject")
    g.to_csv(os.path.join(TAB_PATH, "T1c_fog_cohort.csv"), index=False)
    print(g.to_string(index=False))
    print("windows", X.shape, "freeze prevalence", y.mean().round(4))


if __name__ == "__main__":
    main()
