"""Evaluation metrics (window/beat level, subject level) and statistical helpers."""
import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
                             roc_auc_score, confusion_matrix)


def youden_threshold(y, p):
    """Threshold maximising sensitivity + specificity - 1 (computed on training/inner data only)."""
    if len(np.unique(y)) < 2:
        return 0.5
    o = np.argsort(-p)
    ys, ps = y[o], p[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    P, N = ys.sum(), (1 - ys).sum()
    j = tp / P - fp / N
    k = int(np.argmax(j))
    return float(ps[k])


def binary_metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    tn = int(((yh == 0) & (y == 0)).sum()); fp = int(((yh == 1) & (y == 0)).sum())
    out = dict(n=len(y), pos=int(y.sum()))
    out["AUROC"] = roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
    out["AUPRC"] = average_precision_score(y, p) if y.sum() > 0 else np.nan
    out["Sens"] = tp / (tp + fn) if tp + fn else np.nan
    out["Spec"] = tn / (tn + fp) if tn + fp else np.nan
    out["PPV"] = tp / (tp + fp) if tp + fp else np.nan
    out["F1"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    out["BalAcc"] = np.nanmean([out["Sens"], out["Spec"]])
    return out


def multiclass_metrics(y, P, classes):
    yh = P.argmax(1)
    labels = np.arange(len(classes))
    out = dict(n=len(y), Acc=float((yh == y).mean()),
               MacroF1=f1_score(y, yh, labels=labels[np.isin(labels, y)], average="macro", zero_division=0),
               BalAcc=balanced_accuracy_score(y, yh), Kappa=cohen_kappa_score(y, yh))
    f1s = f1_score(y, yh, labels=labels, average=None, zero_division=0)
    for c, f in zip(classes, f1s):
        out[f"F1_{c}"] = f if (y == list(classes).index(c)).any() else np.nan
    aucs = [roc_auc_score(y == c, P[:, c]) for c in labels if 0 < (y == c).sum() < len(y)]
    out["MacroAUROC"] = float(np.mean(aucs)) if aucs else np.nan
    return out


def cm(y, yh, K):
    return confusion_matrix(y, yh, labels=np.arange(K))


def cluster_bootstrap(groups, fn, n_boot=1000, seed=0):
    """Percentile CI of a statistic fn(index_array) resampling whole subjects (clusters)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        s = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx_by[g] for g in s])
        try:
            v = fn(ii)
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def wilcoxon_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.allclose(a[m], b[m]):
        return np.nan
    return float(stats.wilcoxon(a[m], b[m], zero_method="wilcox").pvalue)


def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    q = p[ok][idx]
    m = len(q)
    adj = np.maximum.accumulate((m - np.arange(m)) * q)
    res = np.empty(m)
    res[idx] = np.minimum(adj, 1.0)
    out[ok] = res
    return out
