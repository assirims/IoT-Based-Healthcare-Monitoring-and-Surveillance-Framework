"""
Central configuration of the IoT-CARES pipeline.

>>> THE ONLY PLACE YOU NEED TO EDIT <<<
Set the three paths below (or the environment variables CARES_DATA, CARES_PROCESSED, CARES_RESULTS).
By default everything lives inside this folder, so the package runs out-of-the-box after unzipping.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------------------------
# 1) PATHS  (edit here)
# ---------------------------------------------------------------------------------------------
DATA_PATH = os.environ.get("CARES_DATA", os.path.join(ROOT, "data", "raw"))            # raw downloaded files
PROCESSED_PATH = os.environ.get("CARES_PROCESSED", os.path.join(ROOT, "data", "processed"))  # windows / features
RESULTS_PATH = os.environ.get("CARES_RESULTS", os.path.join(ROOT, "results"))          # tables, figures, logs

TAB_PATH = os.path.join(RESULTS_PATH, "tables")
FIG_PATH = os.path.join(RESULTS_PATH, "figures")
LOG_PATH = os.path.join(RESULTS_PATH, "logs")
PRED_PATH = os.path.join(RESULTS_PATH, "predictions")
for _p in (PROCESSED_PATH, TAB_PATH, FIG_PATH, LOG_PATH, PRED_PATH):
    os.makedirs(_p, exist_ok=True)

# ---------------------------------------------------------------------------------------------
# 2) RAW FILES (names expected inside DATA_PATH)
# ---------------------------------------------------------------------------------------------
RAW = {
    "fog": "daphnet_fog.zip",                       # UCI Daphnet Freezing of Gait
    "har70": "har70plus.zip",                       # UCI HAR70+
    "falls": "falls_selected.npz",                  # UCI Simulated Falls & ADL (sensor subset, see 01_download_data.py)
    "falls_zip": "simulated_falls_adl.zip",         # (optional) the full 1.2 GB UCI archive
    "ecg": "mitbih.zip",                            # PhysioNet MIT-BIH Arrhythmia Database
    "cog": "casas_cognitive_assessment.zip",        # CASAS cognitive assessment (Zenodo 15713579)
}

URLS = {
    "fog": "https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip",
    "har70": "https://archive.ics.uci.edu/static/public/780/har70.zip",
    "falls_zip": "https://archive.ics.uci.edu/static/public/455/simulated+falls+and+daily+living+activities+data+set.zip",
    "ecg": "https://physionet.org/content/mitdb/get-zip/1.0.0/",
    "cog": "https://zenodo.org/records/15713579/files/cognitive_assessment.zip?download=1",
}

# ---------------------------------------------------------------------------------------------
# 3) EXPERIMENT SETTINGS
# ---------------------------------------------------------------------------------------------
SEED = 2026
SEEDS = [11, 23, 47]                 # repeated runs of stochastic models (MLP / federated)
N_JOBS = int(os.environ.get("CARES_JOBS", "2"))

# windowing per monitoring service (seconds)
WIN = {
    "fog":   dict(fs=64, win_s=4.0, step_s=1.0),
    "har70": dict(fs=50, win_s=5.0, step_s=2.5),
    "falls": dict(fs=25, win_s=4.0, step_s=1.0),
}

# MiniRocket-lite random-kernel transform
ROCKET = dict(dilations=(1, 2, 4, 8, 16), n_bias=2)

# hist-gradient-boosting default (selected on inner validation folds, see 07_train_evaluate.py)
HGB = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0)

# federated learning
FL = dict(rounds=40, local_epochs=1, lr=2e-3, batch=256, hidden=(128, 64), mu=0.01, dropout=0.1)

# personalisation: minutes of labelled data from the new user ("onboarding session")
ONBOARD_MIN = (0, 1, 2, 5)

# calibrated alarm management
ALARM = dict(
    delta=0.10,                       # P(new user exceeds the false-alarm budget) <= delta
    budget_per_h={"fog": 2.0, "falls": 1.0 / 8.0, "har70": 1.0, "ecg": 6.0},
    smooth_k={"fog": 3, "falls": 1, "har70": 3, "ecg": 5},
    refractory_s={"fog": 10.0, "falls": 30.0, "har70": 60.0, "ecg": 60.0},
)
