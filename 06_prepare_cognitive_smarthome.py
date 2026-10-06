
import os
import re
import zipfile

import numpy as np
import pandas as pd

from config import DATA_PATH, PROCESSED_PATH, RAW, TAB_PATH

DX = {1: "dementia", 2: "MCI", 3: "middle-aged", 4: "young-old", 5: "old-old", 6: "other medical",
      7: "at risk", 8: "younger adult", 9: "younger adult (ESL)", 10: "n/a"}


def parse_events(txt):
    rows = []
    for line in txt.splitlines():
        p = line.split()
        mt = re.match(r"(\d\d):(\d\d):(\d\d(?:\.\d{1,6})?)", p[0]) if p else None
        if len(p) < 3 or not mt:
            continue
        t = int(mt.group(1)) * 3600 + int(mt.group(2)) * 60 + float(mt.group(3))
        rows.append((t, p[1], p[2], p[3] if len(p) > 3 else ""))
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=["t", "sensor", "value", "ann"])
    d["t"] = d["t"] - d["t"].iloc[0]
    d.loc[d["t"] < -1, "t"] += 86400            # sessions crossing midnight
    return d


def task_bounds(d):
    b = {}
    for t, ann in zip(d["t"], d["ann"]):
        for tok in ann.split(","):
            m = re.match(r"(\d+)-(start|end)$", tok.strip())
            if m:
                k = int(m.group(1))
                b.setdefault(k, [None, None])
                if m.group(2) == "start" and b[k][0] is None:
                    b[k][0] = t
                elif m.group(2) == "end":
                    b[k][1] = t
    return {k: v for k, v in b.items() if v[0] is not None and v[1] is not None and v[1] > v[0]}


def seg_features(e, dur):
    mot = e[(e.sensor.str[0] == "M") & (e.value == "ON")]
    seq = mot.sensor.to_numpy()
    switches = int((seq[1:] != seq[:-1]).sum()) if len(seq) > 1 else 0
    distinct = len(set(seq))
    gaps = np.diff(e.t.to_numpy()) if len(e) > 1 else np.array([dur])
    return dict(dur=dur, events=len(e), motion=len(mot), distinct_motion=distinct, switches=switches,
                revisit=switches / max(distinct, 1), items=int((e.sensor.str[0] == "I").sum()),
                doors=int((e.sensor.str[0] == "D").sum()), rate=len(e) / max(dur, 1.0),
                max_gap=float(gaps.max()) if len(gaps) else dur, mean_gap=float(gaps.mean()) if len(gaps) else dur)


def main():
    z = zipfile.ZipFile(os.path.join(DATA_PATH, RAW["cog"]))
    root = "cognitive_assessment"
    dx = pd.read_csv(z.open(f"{root}/documents/diagnosis.txt"), sep=r"\s+", skiprows=lambda i: i < 14,
                     names=["id", "dx"], engine="python")
    dx = dx[pd.to_numeric(dx.id, errors="coerce").notna()].astype(int)
    sc_lines = z.read(f"{root}/documents/activityscores.txt").decode("latin1").splitlines()
    scores = {}
    for l in sc_lines:
        p = l.split()
        if len(p) == 12 and p[0].isdigit():
            v = [np.nan if x == "?" else float(x) for x in p[1:]]
            scores[int(p[0])] = dict(score_sum_1_8=v[8], dayout_accuracy=v[9], dayout_sequencing=v[10])
    rows = []
    for n in sorted(x for x in z.namelist() if x.startswith(f"{root}/data/") and x.endswith(".txt")):
        pid = int(os.path.basename(n)[:-4])
        d = parse_events(z.read(n).decode("latin1"))
        if d is None or len(d) < 50:
            continue
        b = task_bounds(d)
        r = dict(id=pid, n_tasks=len(b))
        for k in range(1, 25):
            if k in b:
                s, e_ = b[k]
                f = seg_features(d[(d.t >= s) & (d.t <= e_)], e_ - s)
            else:
                f = {kk: np.nan for kk in ["dur", "events", "motion", "distinct_motion", "switches", "revisit",
                                          "items", "doors", "rate", "max_gap", "mean_gap"]}
            r.update({f"t{k:02d}_{kk}": v for kk, v in f.items()})
        for name, ks in [("uncued", range(1, 9)), ("cued", range(9, 17)), ("dayout", range(17, 25))]:
            ks = [k for k in ks if k in b]
            if ks:
                s = min(b[k][0] for k in ks); e_ = max(b[k][1] for k in ks)
                f = seg_features(d[(d.t >= s) & (d.t <= e_)], e_ - s)
                r.update({f"{name}_{kk}": v for kk, v in f.items()})
        rows.append(r)
    F = pd.DataFrame(rows).merge(dx, on="id", how="left")
    F["group"] = F.dx.map(DX)
    F = F.merge(pd.DataFrame.from_dict(scores, orient="index").rename_axis("id").reset_index(), on="id", how="left")
    F.to_csv(os.path.join(PROCESSED_PATH, "cog_features.csv"), index=False)
    t = F.groupby("group").agg(participants=("id", "size"), tasks_completed_median=("n_tasks", "median"),
                               uncued_minutes_median=("uncued_dur", lambda x: np.nanmedian(x) / 60)).reset_index()
    t.to_csv(os.path.join(TAB_PATH, "T1e_cognitive_cohort.csv"), index=False)
    print(t.to_string(index=False))
    print("participants with sensor data", len(F))


if __name__ == "__main__":
    main()
