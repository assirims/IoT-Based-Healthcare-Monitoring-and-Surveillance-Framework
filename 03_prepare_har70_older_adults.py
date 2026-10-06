
import os
import zipfile

import numpy as np
import pandas as pd

from config import DATA_PATH, PROCESSED_PATH, RAW, WIN, TAB_PATH
from lib.windows import sliding_windows, majority

RAW_TO_CLASS = {1: 0, 3: 1, 4: 2, 5: 2, 6: 3, 7: 4, 8: 5}
CLASSES = ["walking", "shuffling", "stairs", "standing", "sitting", "lying"]
CHANNELS = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]


def main():
    cfg = WIN["har70"]
    z = zipfile.ZipFile(os.path.join(DATA_PATH, RAW["har70"]))
    Xs, ys, subj, ts, pur, rows = [], [], [], [], [], []
    for n in sorted(x for x in z.namelist() if x.endswith(".csv")):
        sid = int(os.path.basename(n)[:-4])
        d = pd.read_csv(z.open(n))
        tt = pd.to_datetime(d["timestamp"]).astype("int64").to_numpy() / 1e9
        sig = d[CHANNELS].to_numpy(np.float32)
        lab = d["label"].map(RAW_TO_CLASS).to_numpy()
        # split at gaps > 1 s so windows never bridge interrupted recordings
        cut = np.where(np.diff(tt) > 1.0)[0] + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(tt)]):
            X, Y, st = sliding_windows(sig[a:b], lab[a:b], cfg["fs"], cfg["win_s"], cfg["step_s"], t0=tt[a] - tt[0])
            if len(X) == 0:
                continue
            m, p, _ = majority(Y)
            Xs.append(X); ys.append(m.astype(np.int8)); pur.append(p.astype(np.float32))
            subj.append(np.full(len(X), sid, np.int16)); ts.append(st)
        r = dict(subject=sid, minutes=round(len(d) / cfg["fs"] / 60, 1))
        for k, c in enumerate(CLASSES):
            r[c + "_min"] = round((lab == k).sum() / cfg["fs"] / 60, 2)
        rows.append(r)
    out = dict(X=np.concatenate(Xs), y=np.concatenate(ys), subject=np.concatenate(subj), t=np.concatenate(ts),
               purity=np.concatenate(pur), classes=np.array(CLASSES), channels=np.array(CHANNELS), fs=cfg["fs"])
    np.savez_compressed(os.path.join(PROCESSED_PATH, "har70_windows.npz"), **out)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB_PATH, "T1b_har70_cohort.csv"), index=False)
    print(t.to_string(index=False))
    print("windows", out["X"].shape, "class counts", np.bincount(out["y"]))


if __name__ == "__main__":
    main()
