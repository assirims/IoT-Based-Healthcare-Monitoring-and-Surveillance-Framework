
import json
import os
import time

import numpy as np

from config import PROCESSED_PATH, TAB_PATH
from lib.features import extract

FALL_SITES = ["waist", "wrist", "chest", "thigh", "ankle", "head"]


def sensors_for(service):
    if service == "fog":
        return {"ankle": [0, 1, 2], "thigh": [3, 4, 5], "trunk": [6, 7, 8]}
    if service == "har70":
        return {"back": [0, 1, 2], "thigh": [3, 4, 5]}
    if service == "falls":
        s = {}
        for k, site in enumerate(FALL_SITES):
            s[f"{site}_acc"] = [6 * k, 6 * k + 1, 6 * k + 2]
            s[f"{site}_gyr"] = [6 * k + 3, 6 * k + 4, 6 * k + 5]
        return s
    raise ValueError(service)


# default sensor kit of every service (what a person would actually wear)
KIT = {"fog": ["ankle", "thigh", "trunk"], "har70": ["back", "thigh"],
       "falls": ["waist_acc", "waist_gyr", "wrist_acc", "wrist_gyr"]}


def main():
    timing = {}
    for service in ["fog", "har70", "falls"]:
        d = np.load(os.path.join(PROCESSED_PATH, f"{service}_windows.npz"))
        X, fs = d["X"], float(d["fs"])
        sens = sensors_for(service)
        t0 = time.perf_counter()
        F, names = extract(X, fs, sens)
        dt = time.perf_counter() - t0
        # single-window latency (what one gateway call costs)
        one = X[:1]
        kit = {k: sens[k] for k in KIT[service]}
        reps = 200
        t1 = time.perf_counter()
        for _ in range(reps):
            extract(one, fs, kit)
        timing[service] = dict(windows=len(X), batch_ms_per_window=1000 * dt / len(X),
                               single_window_ms_kit=1000 * (time.perf_counter() - t1) / reps,
                               n_features_all=len(names), n_features_kit=sum(1 for n in names if n.split(":")[0] in KIT[service]))
        meta = {k: d[k] for k in d.files if k != "X"}
        np.savez_compressed(os.path.join(PROCESSED_PATH, f"{service}_features.npz"), F=F, names=np.array(names), **meta)
        print(service, F.shape, timing[service], flush=True)
    json.dump(timing, open(os.path.join(TAB_PATH, "S1_feature_extraction_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
