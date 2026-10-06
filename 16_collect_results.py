
import glob
import json
import os

import numpy as np
import pandas as pd

from config import TAB_PATH, RESULTS_PATH, PROCESSED_PATH, DATA_PATH, RAW


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def cohort():
    c = {}
    t = pd.read_csv(os.path.join(TAB_PATH, "T1a_falls_cohort.csv"))
    c["falls"] = dict(people=len(t), trials=int(t.trials.sum()), fall_trials=int(t.falls.sum()),
                      adl_hours=float(t.adl_minutes.sum() / 60), age_min=int(t.age.min()), age_max=int(t.age.max()),
                      women=int((t.sex.str.lower() == "female").sum()))
    t = pd.read_csv(os.path.join(TAB_PATH, "T1b_har70_cohort.csv"))
    w = np.load(os.path.join(PROCESSED_PATH, "har70_windows.npz"))
    c["har70"] = dict(people=len(t), hours=float(t.minutes.sum() / 60), windows=int(len(w["y"])),
                      class_windows={k: int(v) for k, v in zip(w["classes"], np.bincount(w["y"], minlength=len(w["classes"])))})
    t = pd.read_csv(os.path.join(TAB_PATH, "T1c_fog_cohort.csv"))
    c["fog"] = dict(people=len(t), hours=float(t.minutes_protocol.sum() / 60), episodes=int(t.n_freeze_episodes.sum()),
                    freeze_minutes=float(t.minutes_freeze.sum()), windows=int(t.windows.sum()),
                    prevalence=float(t.freeze_windows.sum() / t.windows.sum()), non_freezers=int((t.n_freeze_episodes == 0).sum()))
    t = pd.read_csv(os.path.join(TAB_PATH, "T1d_ecg_cohort.csv"))
    g = t.groupby("set")[["beats", "N", "S", "V", "F", "Q"]].sum()
    c["ecg"] = dict(records=len(t), ds1_beats=int(g.loc["DS1", "beats"]), ds2_beats=int(g.loc["DS2", "beats"]),
                    by_set={s: {k: int(v) for k, v in g.loc[s].items()} for s in g.index})
    f = pd.read_csv(os.path.join(PROCESSED_PATH, "cog_features.csv"))
    gc = f.group.value_counts()
    c["cog"] = dict(with_data=len(f), dementia=int(gc.get("dementia", 0)), mci=int(gc.get("MCI", 0)),
                    impaired=int(gc.get("dementia", 0) + gc.get("MCI", 0)),
                    healthy=int(gc.get("middle-aged", 0) + gc.get("young-old", 0) + gc.get("old-old", 0)),
                    healthy60=int(gc.get("young-old", 0) + gc.get("old-old", 0)), groups={k: int(v) for k, v in gc.items()})
    # MIT-BIH: 44 non-paced records from 43 people (records 201 and 202 come from the same person)
    c["total_people"] = c["falls"]["people"] + c["har70"]["people"] + c["fog"]["people"] + 43 + c["cog"]["with_data"]
    return c


def main():
    N = {"cohort": cohort(), "tables": {}}
    for p in sorted(glob.glob(os.path.join(TAB_PATH, "*.csv"))):
        name = os.path.basename(p)[:-4]
        N["tables"][name] = pd.read_csv(p).to_dict(orient="records")
    for p in sorted(glob.glob(os.path.join(TAB_PATH, "*.json"))):
        N["tables"][os.path.basename(p)[:-5]] = json.load(open(p))
    json.dump(clean(N), open(os.path.join(RESULTS_PATH, "paper_numbers.json"), "w"), indent=1)
    print("wrote paper_numbers.json with", len(N["tables"]), "tables; total people", N["cohort"]["total_people"])


if __name__ == "__main__":
    main()
