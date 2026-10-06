"""
Calibrated alarm manager: turns a stream of window probabilities into caregiver alerts with a false-alarm budget.

1) causal smoothing of the score (moving average of the last k windows),
2) alert when the smoothed score crosses the threshold, followed by a refractory period,
3) threshold chosen by SUBJECT-LEVEL CONFORMAL CALIBRATION:
   for every calibration person j compute theta_j = the smallest threshold whose false-alarm rate on j's
   out-of-fold scores is <= B (alarms per hour). With n exchangeable calibration people,
   theta_hat = the ceil((n+1)(1-delta))-th smallest theta_j guarantees
        P( false-alarm rate of a new person <= B ) >= 1 - delta
   because the false-alarm rate is non-increasing in the threshold (we use its monotone envelope).
"""
import numpy as np


def smooth(p, k):
    if k <= 1:
        return p
    c = np.cumsum(np.r_[0.0, p])
    out = np.empty_like(p, dtype=float)
    for i in range(len(p)):
        a = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[a]) / (i + 1 - a)
    return out


def alarms(times, score, thr, refractory):
    """times sorted (alarm time = end of the window). Returns alarm times."""
    out, last = [], -np.inf
    for t, s in zip(times, score):
        if s >= thr and t - last >= refractory:
            out.append(t)
            last = t
    return np.array(out)


def conformal_threshold(theta_j, delta):
    """Finite-sample conformal quantile of the per-person thresholds (returns +inf if n is too small)."""
    th = np.sort(np.asarray(theta_j, float))
    n = len(th)
    r = int(np.ceil((n + 1) * (1 - delta)))
    if r > n:
        return np.inf
    return th[r - 1]


def min_threshold_for_budget(fa_rate_fn, grid, budget):
    """Smallest grid threshold t such that the false-alarm rate is <= budget for t and every larger grid value."""
    rates = np.array([fa_rate_fn(t) for t in grid])
    ok = rates <= budget + 1e-12
    # monotone envelope: t is admissible only if all larger thresholds are admissible as well
    adm = np.flip(np.logical_and.accumulate(np.flip(ok)))
    idx = np.where(adm)[0]
    return grid[idx[0]] if len(idx) else grid[-1]


THRESH_GRID = np.r_[np.linspace(0.0, 0.9, 91), np.linspace(0.901, 1.0, 100)]
