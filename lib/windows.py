"""Sliding-window segmentation shared by the wearable monitoring services."""
import numpy as np


def sliding_windows(sig, lab, fs, win_s, step_s, t0=0.0):
    """Cut a continuous recording into overlapping windows.

    sig : (T, C) float array, lab : (T,) int array (per-sample label)
    returns X (N, L, C) float32, label matrix (N, L) int16, start times (N,) seconds
    """
    L = int(round(win_s * fs))
    S = int(round(step_s * fs))
    T = len(sig)
    if T < L:
        return (np.zeros((0, L, sig.shape[1]), np.float32), np.zeros((0, L), np.int16), np.zeros(0))
    starts = np.arange(0, T - L + 1, S)
    idx = starts[:, None] + np.arange(L)[None, :]
    X = sig[idx].astype(np.float32)
    Y = lab[idx].astype(np.int16)
    return X, Y, t0 + starts / fs


def majority(Y, ignore=None):
    """Majority label of every window (rows of Y) and its purity; `ignore` labels are not counted."""
    labs = np.unique(Y)
    if ignore is not None:
        labs = labs[~np.isin(labs, ignore)]
    counts = np.stack([(Y == l).sum(1) for l in labs], 1)
    best = counts.argmax(1)
    valid = counts.sum(1)
    purity = counts.max(1) / np.maximum(valid, 1)
    return labs[best], purity, valid / Y.shape[1]
