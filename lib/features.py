"""
Edge feature extractor shared by all wearable services (fully vectorised NumPy; runs on a gateway CPU).

For every sensor (3 axes) the extractor uses four signals - the three axes and the vector magnitude - and
computes 22 descriptors per signal (time domain, spectral, gait regularity), plus three inter-axis correlations
per sensor. All operations are causal within the window, so a window can be scored as soon as it is complete.
"""
import numpy as np

STAT_NAMES = ["mean", "std", "min", "max", "p10", "p50", "p90", "iqr", "skew", "kurt", "rms", "jerk",
              "zcr", "dom_freq", "spec_entropy", "bp_0.5_3", "bp_3_8", "bp_8_12", "freeze_index",
              "total_power", "acf_peak", "acf_lag"]


def _moments(x):
    m = x.mean(-1, keepdims=True)
    d = x - m
    s = d.std(-1) + 1e-8
    skew = (d ** 3).mean(-1) / s ** 3
    kurt = (d ** 4).mean(-1) / s ** 4 - 3.0
    return m[..., 0], s, skew, kurt, d


def signal_stats(x, fs):
    """x: (N, S, L) -> (N, S, 22)"""
    N, S, L = x.shape
    mean, std, skew, kurt, d = _moments(x)
    q = np.percentile(x, [10, 50, 90, 25, 75], axis=-1)
    rms = np.sqrt((x ** 2).mean(-1))
    jerk = np.abs(np.diff(x, axis=-1)).mean(-1) * fs
    zcr = (np.diff(np.signbit(d).astype(np.int8), axis=-1) != 0).mean(-1)
    win = np.hanning(L).astype(np.float32)
    P = np.abs(np.fft.rfft(d * win, axis=-1)) ** 2
    f = np.fft.rfftfreq(L, 1.0 / fs)
    P[..., 0] = 0.0
    tot = P.sum(-1) + 1e-12
    pn = P / tot[..., None]
    dom = f[P.argmax(-1)]
    ent = -(pn * np.log(pn + 1e-12)).sum(-1) / np.log(P.shape[-1])

    def band(a, b):
        return P[..., (f >= a) & (f < b)].sum(-1)

    b1, b2, b3 = band(0.5, 3.0), band(3.0, 8.0), band(8.0, min(12.0, fs / 2))
    fi = np.log((b2 + 1e-6) / (b1 + 1e-6))
    # autocorrelation peak in the gait-cycle range 0.4-2.0 s (step/stride regularity)
    F = np.fft.rfft(d, n=2 * L, axis=-1)
    ac = np.fft.irfft(np.abs(F) ** 2, axis=-1)[..., :L]
    ac = ac / (ac[..., :1] + 1e-12)
    lo, hi = int(0.4 * fs), min(int(2.0 * fs), L - 1)
    seg = ac[..., lo:hi]
    acf_peak = seg.max(-1)
    acf_lag = (seg.argmax(-1) + lo) / fs
    feats = [mean, std, x.min(-1), x.max(-1), q[0], q[1], q[2], q[4] - q[3], skew, kurt, rms, jerk, zcr,
             dom, ent, b1 / tot, b2 / tot, b3 / tot, fi, np.log(tot), acf_peak, acf_lag]
    return np.stack(feats, -1).astype(np.float32)


def extract(X, fs, sensors, chunk=4000):
    """X: (N, L, C); sensors: dict name -> 3 channel indices. Returns (F (N, D), names)."""
    out = []
    for a in range(0, len(X), chunk):
        xb = X[a:a + chunk].astype(np.float32)
        blocks = []
        for nm, ch in sensors.items():
            ax = np.transpose(xb[:, :, ch], (0, 2, 1))                       # (n, 3, L)
            mag = np.sqrt((ax ** 2).sum(1, keepdims=True))
            sig = np.concatenate([ax, mag], 1)                              # (n, 4, L)
            st = signal_stats(sig, fs).reshape(len(xb), -1)
            c = _axis_corr(ax)
            sma = np.abs(ax - ax.mean(-1, keepdims=True)).sum(1).mean(-1, keepdims=True)
            blocks += [st, c, sma]
        out.append(np.concatenate(blocks, 1))
    names = []
    for nm in sensors:
        for sname in ["x", "y", "z", "mag"]:
            names += [f"{nm}:{sname}:{s}" for s in STAT_NAMES]
        names += [f"{nm}:corr_xy", f"{nm}:corr_xz", f"{nm}:corr_yz", f"{nm}:sma"]
    F = np.concatenate(out, 0)
    F[~np.isfinite(F)] = 0.0
    return F, names


def _axis_corr(ax):
    d = ax - ax.mean(-1, keepdims=True)
    s = np.sqrt((d ** 2).sum(-1)) + 1e-8
    c = lambda i, j: (d[:, i] * d[:, j]).sum(-1) / (s[:, i] * s[:, j])
    return np.stack([c(0, 1), c(0, 2), c(1, 2)], 1)


def sensor_of(names):
    """Sensor (body location) of each feature name."""
    return np.array([n.split(":")[0] for n in names])
