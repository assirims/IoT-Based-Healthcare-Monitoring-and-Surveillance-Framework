
import io
import os
import re
import zipfile

import numpy as np
import pandas as pd

from config import DATA_PATH, PROCESSED_PATH, RAW, TAB_PATH, WIN

SENSOR_IDS = {"340506": "head", "340527": "chest", "340535": "waist", "340537": "wrist",
              "340539": "thigh", "340540": "ankle"}
SENSOR_ORDER = ["waist", "wrist", "chest", "thigh", "ankle", "head"]
FALL_CODES = list(range(901, 921))
ADL_CODES = list(range(801, 817))
ACT_NAMES = {
    801: "walking-fw", 802: "walking-bw", 803: "jogging", 804: "squatting-down", 805: "bending",
    806: "bending-pick-up", 807: "limp", 808: "stumble", 809: "trip-over", 810: "coughing-sneezing",
    811: "sit-chair", 812: "sit-sofa", 813: "sit-air", 814: "sit-bed", 815: "lying-bed", 816: "rising-bed",
    901: "front-lying", 902: "front-protecting-lying", 903: "front-knees", 904: "front-knees-lying",
    905: "front-quick-recovery", 906: "front-slow-recovery", 907: "front-right", 908: "front-left",
    909: "back-sitting", 910: "back-lying", 911: "back-right", 912: "back-left", 913: "right-sideway",
    914: "right-recovery", 915: "left-sideway", 916: "left-recovery", 917: "rolling-out-bed", 918: "podium",
    919: "syncope", 920: "syncope-wall"}
COLS = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]


def parse_sensor_txt(b):
    df = pd.read_csv(io.BytesIO(b), sep="\t", comment="/", usecols=lambda c: c in COLS + ["Counter"])
    return df["Counter"].to_numpy(), df[COLS].to_numpy(np.float32)


def stage_a(rar_path, out_path):
    from lib.archive import iter_archive
    recs, people = [], []
    for inner_name, inner in iter_archive(path=rar_path, want=lambda n: n.endswith(".rar")):
        sid = int(re.findall(r"(\d{3})\.rar", inner_name)[0])
        trials, errs = {}, []
        for name, data in iter_archive(data=inner, errors=errs):
            parts = name.split("/")
            if name.endswith("Bilgi.txt"):
                txt = data.decode("latin1")
                g = {k.strip().lower(): v.strip() for k, v in (l.split(":", 1) for l in txt.splitlines() if ":" in l)}
                # keep only non-identifying descriptors
                people.append(dict(subject=sid, sex=g.get("gender", g.get("cinsiyet", "")),
                                   age=re.sub(r"\D", "", g.get("age", g.get("yaş", ""))),
                                   height_cm=re.sub(r"\D", "", g.get("height", g.get("boy", ""))),
                                   weight_kg=re.sub(r"\D", "", g.get("weight", g.get("kilo", "")))))
                continue
            if len(parts) < 5 or not parts[-1].endswith(".txt"):
                continue
            repstr = parts[-2].split("_")[-1]
            if not repstr.isdigit():                 # e.g. "Test_3Fail": trial marked as failed by the authors
                continue
            act = int(parts[-3].split("-")[0])
            rep = int(repstr)
            sensor = SENSOR_IDS.get(parts[-1][:-4])
            if sensor is None:
                continue
            try:
                cnt, val = parse_sensor_txt(data)
            except Exception:
                continue
            trials.setdefault((act, rep), {})[sensor] = (cnt, val)
        for (act, rep), d in sorted(trials.items()):
            if any(s not in d for s in SENSOR_ORDER):
                continue
            # align the six units on the common sample counter
            common = set(d[SENSOR_ORDER[0]][0])
            for s in SENSOR_ORDER[1:]:
                common &= set(d[s][0])
            common = np.array(sorted(common))
            if len(common) < 50:
                continue
            arr = []
            for s in SENSOR_ORDER:
                cnt, val = d[s]
                _, ia, _ = np.intersect1d(cnt, common, return_indices=True)
                arr.append(val[ia])
            sig = np.concatenate(arr, 1)                                  # (T, 36)
            if not np.isfinite(sig).all():
                sig = pd.DataFrame(sig).interpolate(limit_direction="both").to_numpy(np.float32)
            recs.append((sid, act, rep, sig))
        print("subject", sid, "trials", sum(1 for r in recs if r[0] == sid), "unreadable entries", len(errs), flush=True)
    lens = np.array([len(r[3]) for r in recs])
    off = np.r_[0, np.cumsum(lens)]
    np.savez_compressed(out_path, signal=np.concatenate([r[3] for r in recs]).astype(np.float32), offsets=off,
                        subject=np.array([r[0] for r in recs]), activity=np.array([r[1] for r in recs]),
                        repetition=np.array([r[2] for r in recs]), sensors=np.array(SENSOR_ORDER),
                        channels=np.array([f"{s}_{c}" for s in SENSOR_ORDER for c in COLS]), fs=25.0,
                        people=pd.DataFrame(people).to_json())
    print("wrote", out_path, len(recs), "trials")


def stage_b():
    from lib.falls_io import load
    d = load()
    sig, off = d["signal"], d["offsets"]
    subj, act, rep = d["subject"], d["activity"], d["repetition"]
    fs = int(d["fs"])
    cfg = WIN["falls"]
    L = int(cfg["win_s"] * fs)
    Xs, rows = [], []
    for i in range(len(subj)):
        s = sig[off[i]:off[i + 1]]
        # peak of the waist acceleration magnitude locates the event; the window covers 2 s before / 2 s after
        mag = np.linalg.norm(s[:, 0:3], axis=1)
        p = int(np.argmax(mag))
        a = int(np.clip(p - L // 2, 0, max(len(s) - L, 0)))
        w = s[a:a + L]
        if len(w) < L:
            w = np.pad(w, ((0, L - len(w)), (0, 0)), mode="edge")
        Xs.append(w)
        rows.append(dict(subject=subj[i], activity=act[i], repetition=rep[i], fall=int(act[i] >= 900),
                         duration_s=len(s) / fs, peak_s=p / fs))
    meta = pd.DataFrame(rows)
    X = np.stack(Xs).astype(np.float32)
    np.savez_compressed(os.path.join(PROCESSED_PATH, "falls_windows.npz"), X=X, y=meta.fall.to_numpy(np.int8),
                        subject=meta.subject.to_numpy(np.int16), activity=meta.activity.to_numpy(np.int16),
                        duration_s=meta.duration_s.to_numpy(np.float32), channels=d["channels"], fs=fs)
    people = pd.read_json(io.StringIO(str(d["people"])))
    t = meta.groupby("subject").agg(trials=("fall", "size"), falls=("fall", "sum"),
                                    adl_minutes=("duration_s", lambda x: 0)).reset_index()
    t["adl_minutes"] = meta[meta.fall == 0].groupby("subject").duration_s.sum().reindex(t.subject).to_numpy() / 60
    t = t.merge(people, on="subject", how="left")
    t.to_csv(os.path.join(TAB_PATH, "T1a_falls_cohort.csv"), index=False)
    print(t.to_string(index=False))
    print("windows", X.shape, "fall prevalence", meta.fall.mean().round(3))


def main():
    out = os.path.join(DATA_PATH, RAW["falls"])
    if not os.path.exists(out) and not os.path.isdir(os.path.join(DATA_PATH, "falls_selected")):
        rar = os.path.join(PROCESSED_PATH, "Tests.rar")
        if not os.path.exists(rar):
            with zipfile.ZipFile(os.path.join(DATA_PATH, RAW["falls_zip"])) as z, open(rar, "wb") as f:
                with z.open("Tests.rar") as src:
                    while True:
                        b = src.read(1 << 24)
                        if not b:
                            break
                        f.write(b)
        stage_a(rar, out)
    stage_b()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--from-rar":
        stage_a(sys.argv[2], os.path.join(DATA_PATH, RAW["falls"]))
    main()
