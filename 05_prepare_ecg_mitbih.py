
import os
import zipfile

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

from config import DATA_PATH, PROCESSED_PATH, RAW, TAB_PATH
from lib.wfdb_lite import AAMI, read_212, read_atr, read_header

DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
PACED = [102, 104, 107, 217]
CLASSES = ["N", "S", "V", "F", "Q"]
PRE, POST, DS = 90, 144, 6                 # 250 ms before / 400 ms after the R peak at 360 Hz, decimate by 6


def preprocess(x, fs):
    base = median_filter(median_filter(x, size=int(0.2 * fs) | 1), size=int(0.6 * fs) | 1)
    y = x - base
    b, a = butter(3, 35.0 / (fs / 2), btype="low")
    return filtfilt(b, a, y)


def beat_features(sig, r, fs):
    rr = np.diff(r) / fs
    pre = np.r_[rr[0], rr]
    post = np.r_[rr, rr[-1]]
    local = pd.Series(pre).rolling(10, min_periods=1).mean().to_numpy()
    t = r / fs
    # causal "global" rhythm: mean RR over the previous 5 minutes
    cs = np.cumsum(pre)
    j = np.searchsorted(t, t - 300.0)
    glob = (cs - np.r_[0, cs][j]) / np.maximum(np.arange(len(r)) - j + 1, 1)
    rrf = np.stack([pre, post, local, glob, pre / glob, post / glob, local / glob, pre / post,
                    (post - pre) / glob], 1)
    segs = np.stack([sig[max(i - PRE, 0): i + POST] if (i - PRE >= 0 and i + POST <= len(sig))
                     else np.pad(sig[max(i - PRE, 0): i + POST], (max(PRE - i, 0), max(i + POST - len(sig), 0)), mode="edge")
                     for i in r])
    morph = segs[:, ::DS]
    raw = segs[:, ::2]                                                      # 180 Hz beat segment (117 samples)
    qrs = segs[:, PRE - 18: PRE + 18]                                       # +-50 ms around R
    extra = np.stack([segs[:, PRE], qrs.min(1), qrs.max(1) - qrs.min(1), (qrs ** 2).sum(1),
                      np.abs(np.diff(qrs, axis=1)).sum(1), segs[:, PRE + 36:].max(1), segs[:, :PRE - 18].max(1),
                      (np.abs(qrs) > 0.5 * np.abs(qrs).max(1, keepdims=True)).sum(1) / fs], 1)
    return rrf.astype(np.float32), morph.astype(np.float32), extra.astype(np.float32), raw.astype(np.float16)


def main():
    z = zipfile.ZipFile(os.path.join(DATA_PATH, RAW["ecg"]))
    names = z.namelist()
    root = names[0].split("/")[0]
    out = {k: [] for k in ["rr", "morph", "extra", "seg", "label", "record", "t", "rhythm"]}
    rows = []
    for rec in sorted(DS1 + DS2):
        h = read_header(z.read(f"{root}/{rec}.hea").decode())
        x = read_212(z.read(f"{root}/{rec}.dat"), h["nsig"])
        ch = [i for i, s in enumerate(h["signals"]) if "MLII" in s["desc"]][0]
        s = h["signals"][ch]
        sig = (x[:, ch].astype(np.float64) - s["adczero"]) / s["gain"]
        sig = preprocess(sig, h["fs"])
        smp, sym, aux = read_atr(z.read(f"{root}/{rec}.atr"))
        # rhythm label in force at every annotation
        rhythm, cur = [], "(N"
        for sy, au in zip(sym, aux):
            if sy == "+" and au:
                cur = au.strip()
            rhythm.append(cur)
        rhythm = np.array(rhythm)
        isbeat = np.isin(sym, list(AAMI.keys()))
        r, lab, rh = smp[isbeat], np.array([AAMI[s_] for s_ in sym[isbeat]]), rhythm[isbeat]
        keep = (r > PRE) & (r < len(sig) - POST)
        # features use all beats (RR context) but beats too close to the borders are dropped afterwards
        rrf, morph, extra, seg = beat_features(sig, r, h["fs"])
        out["rr"].append(rrf[keep]); out["morph"].append(morph[keep]); out["extra"].append(extra[keep])
        out["seg"].append(seg[keep])
        out["label"].append(np.array([CLASSES.index(l) for l in lab[keep]], np.int8))
        out["record"].append(np.full(keep.sum(), rec, np.int16)); out["t"].append((r[keep] / h["fs"]).astype(np.float32))
        out["rhythm"].append(rh[keep])
        cnt = pd.Series(lab[keep]).value_counts()
        rows.append(dict(record=rec, set="DS1" if rec in DS1 else "DS2", beats=int(keep.sum()),
                         **{c: int(cnt.get(c, 0)) for c in CLASSES}, lead=s["desc"]))
    res = {k: np.concatenate(v) for k, v in out.items()}
    np.savez_compressed(os.path.join(PROCESSED_PATH, "ecg_beats.npz"), **res, classes=np.array(CLASSES),
                        ds1=np.array(DS1), ds2=np.array(DS2), fs=360)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB_PATH, "T1d_ecg_cohort.csv"), index=False)
    print(t.groupby("set")[["beats"] + CLASSES].sum())
    print("total beats", len(res["label"]))


if __name__ == "__main__":
    main()
