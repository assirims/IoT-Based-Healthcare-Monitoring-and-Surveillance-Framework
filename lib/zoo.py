"""Data access for the wearable services and the model zoo compared in the benchmark."""
import os
import pickle
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_PATH, ROCKET, HGB
from lib.minirocket import MiniRocketLite
from lib.nn import MLP, train_local

KIT = {"fog": ["ankle", "thigh", "trunk"], "har70": ["back", "thigh"],
       "falls": ["waist_acc", "waist_gyr", "wrist_acc", "wrist_gyr"]}
THR_FEATURE = {"fog": "ankle:y:freeze_index", "falls": "waist_acc:mag:max"}
SERVICE_NAME = {"fog": "Freezing of gait (Parkinson's disease)", "har70": "Activity monitoring (adults 70+)",
                "falls": "Fall detection", "ecg": "Arrhythmia monitoring (ECG patch)"}


def rocket_input(service, X):
    if service == "fog":
        mags = [np.linalg.norm(X[:, :, c:c + 3], axis=2) for c in (0, 3, 6)]
        vert = [X[:, :, c] for c in (1, 4, 7)]
        return np.stack(mags + vert, -1).astype(np.float32)
    if service == "har70":
        mags = [np.linalg.norm(X[:, :, c:c + 3], axis=2) for c in (0, 3)]
        vert = [X[:, :, c] for c in (0, 3)]
        return np.stack(mags + vert, -1).astype(np.float32)
    if service == "falls":
        # waist acc, waist gyr, wrist acc, wrist gyr magnitudes (channel layout: 6 per site, waist first, wrist second)
        return np.stack([np.linalg.norm(X[:, :, c:c + 3], axis=2) for c in (0, 3, 6, 9)], -1).astype(np.float32)
    raise ValueError(service)


def load_service(service, kit_only=True):
    f = np.load(os.path.join(PROCESSED_PATH, f"{service}_features.npz"), allow_pickle=False)
    w = np.load(os.path.join(PROCESSED_PATH, f"{service}_windows.npz"), allow_pickle=False)
    names = f["names"]
    keep = np.isin(np.array([n.split(":")[0] for n in names]), KIT[service]) if kit_only else np.ones(len(names), bool)
    d = dict(F=f["F"][:, keep], names=names[keep], F_all=f["F"], names_all=names, y=f["y"].astype(int),
             subject=f["subject"].astype(int), R=rocket_input(service, w["X"]), fs=float(w["fs"]))
    d["t"] = f["t"] if "t" in f.files else np.zeros(len(d["y"]))
    d["classes"] = list(f["classes"]) if "classes" in f.files else ["no event", "event"]
    d["K"] = len(d["classes"])
    for k in ("activity", "duration_s", "purity"):
        if k in f.files:
            d[k] = f[k]
    return d


# ------------------------------------------------------------------------------------------------------------
class Model:
    name = "base"
    uses = "F"           # which view: F (handcrafted) or M (MiniRocket features)

    def size_bytes(self):
        return len(pickle.dumps(self, protocol=4))


class SingleFeature(Model):
    """Conventional threshold detector (e.g. peak acceleration for falls, freeze index for FoG)."""
    def __init__(self, idx):
        self.idx = idx
        self.name = "Threshold"

    def fit(self, F, y, K):
        self.lr = LogisticRegression(max_iter=1000).fit(F[:, [self.idx]], y)
        return self

    def predict_proba(self, F):
        return self.lr.predict_proba(F[:, [self.idx]])


class LR(Model):
    name = "LR"

    def fit(self, F, y, K):
        self.sc = StandardScaler().fit(F)
        self.m = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000).fit(self.sc.transform(F), y)
        return self

    def predict_proba(self, F):
        return self.m.predict_proba(self.sc.transform(F))


class RF(Model):
    name = "RF"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, F, y, K):
        self.m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                        max_features="sqrt", n_jobs=1, random_state=self.seed).fit(F, y)
        return self

    def predict_proba(self, F):
        return self.m.predict_proba(F)


class GBM(Model):
    name = "HGB"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, F, y, K):
        self.m = HistGradientBoostingClassifier(early_stopping=False, random_state=self.seed, **HGB).fit(F, y)
        return self

    def predict_proba(self, F):
        return self.m.predict_proba(F)


class NeuralMLP(Model):
    name = "MLP"

    def __init__(self, seed=0, epochs=12, hidden=(128, 64)):
        self.seed, self.epochs, self.hidden = seed, epochs, hidden

    def fit(self, F, y, K):
        self.sc = StandardScaler().fit(F)
        Z = np.clip(self.sc.transform(F), -8, 8)
        self.K = K
        self.net = MLP(Z.shape[1], self.hidden, K, seed=self.seed, dropout=0.1)
        opt = None
        for ep in range(self.epochs):
            opt = train_local(self.net, Z, y, K, epochs=1, lr=1e-3, batch=256, wd=1e-5, seed=self.seed + ep, opt=opt)
        return self

    def predict_proba(self, F):
        return self.net.predict_proba(np.clip(self.sc.transform(F), -8, 8))


class RocketLR(Model):
    name = "MiniRocket-LR"
    uses = "M"

    def fit(self, M, y, K):
        self.sc = StandardScaler().fit(M)
        self.m = LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000).fit(self.sc.transform(M), y)
        return self

    def predict_proba(self, M):
        return self.m.predict_proba(self.sc.transform(M))


class CARES(Model):
    """Proposed dual-view edge classifier: boosted trees on physiological descriptors + linear model on
    random-convolution features of the raw signal, fused by probability averaging."""
    name = "CARES (dual-view)"
    uses = "FM"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, F, M, y, K):
        self.a = GBM(self.seed).fit(F, y, K)
        self.b = RocketLR().fit(M, y, K)
        return self

    def predict_proba(self, F, M):
        return 0.5 * (self.a.predict_proba(F) + self.b.predict_proba(M))


def make_models(service, names, seed=0):
    ms = []
    if service in THR_FEATURE:
        ms.append(SingleFeature(int(np.where(np.array(names) == THR_FEATURE[service])[0][0])))
    ms += [LR(), RF(seed), GBM(seed), NeuralMLP(seed), RocketLR(), CARES(seed)]
    return ms


def fit_predict(model, Ftr, Mtr, ytr, Fte, Mte, K):
    t0 = time.perf_counter()
    if model.uses == "FM":
        model.fit(Ftr, Mtr, ytr, K)
        t1 = time.perf_counter()
        p = model.predict_proba(Fte, Mte)
    elif model.uses == "M":
        model.fit(Mtr, ytr, K)
        t1 = time.perf_counter()
        p = model.predict_proba(Mte)
    else:
        model.fit(Ftr, ytr, K)
        t1 = time.perf_counter()
        p = model.predict_proba(Fte)
    t2 = time.perf_counter()
    P = np.zeros((len(p), K))
    cls = getattr(getattr(model, "m", None), "classes_", None)
    if p.shape[1] == K:
        P = p
    else:  # a class absent from the training fold
        present = np.unique(ytr)
        P[:, present] = p
    return P, (t1 - t0), (t2 - t1) / max(len(p), 1)


def make_rocket(service, Rtr, seed=0):
    return MiniRocketLite(dilations=ROCKET["dilations"][:4], n_bias=1, seed=seed).fit(Rtr)
