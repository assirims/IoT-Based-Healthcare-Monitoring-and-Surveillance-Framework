"""Read/write the accelerometer+gyroscope subset of the UCI falls dataset (single file or one file per volunteer)."""
import glob
import os

import numpy as np

from config import DATA_PATH, RAW

DIR = os.path.join(DATA_PATH, "falls_selected")


def load():
    single = os.path.join(DATA_PATH, RAW["falls"])
    if os.path.exists(single):
        d = np.load(single, allow_pickle=False)
        return {k: d[k] for k in d.files}
    files = sorted(glob.glob(os.path.join(DIR, "falls_*.npz")))
    if not files:
        raise FileNotFoundError("falls subset not found: run 04_prepare_falls.py with the UCI archive (see README)")
    parts = [np.load(f, allow_pickle=False) for f in files]
    sig = np.concatenate([p["signal"] for p in parts])
    lens = np.concatenate([np.diff(p["offsets"]) for p in parts])
    out = dict(signal=sig, offsets=np.r_[0, np.cumsum(lens)],
               subject=np.concatenate([p["subject"] for p in parts]), activity=np.concatenate([p["activity"] for p in parts]),
               repetition=np.concatenate([p["repetition"] for p in parts]), sensors=parts[0]["sensors"],
               channels=parts[0]["channels"], fs=parts[0]["fs"])
    import pandas as pd, io
    out["people"] = pd.concat([pd.read_json(io.StringIO(str(p["people"]))) for p in parts]).to_json()
    return out


def split_per_subject():
    """Write one compressed file per volunteer (keeps every file small enough for e-mail / repositories)."""
    d = load()
    os.makedirs(DIR, exist_ok=True)
    import pandas as pd, io
    ppl = pd.read_json(io.StringIO(str(d["people"])))
    off = d["offsets"]
    for s in np.unique(d["subject"]):
        idx = np.where(d["subject"] == s)[0]
        segs = [d["signal"][off[i]:off[i + 1]] for i in idx]
        lens = np.array([len(x) for x in segs])
        np.savez_compressed(os.path.join(DIR, f"falls_{s}.npz"), signal=np.concatenate(segs), offsets=np.r_[0, np.cumsum(lens)],
                            subject=d["subject"][idx], activity=d["activity"][idx], repetition=d["repetition"][idx],
                            sensors=d["sensors"], channels=d["channels"], fs=d["fs"],
                            people=ppl[ppl.subject == s].to_json())
    print("wrote", len(np.unique(d["subject"])), "files to", DIR)
