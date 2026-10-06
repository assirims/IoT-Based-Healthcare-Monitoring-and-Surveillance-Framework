"""
MiniRocket-lite: a training-free random-convolution transform for raw sensor windows (after Dempster et al.,
KDD 2021), written in NumPy so that it can run on an IoT gateway without a deep-learning runtime.

84 fixed kernels of length 9 with weights in {-1, 2} (three positions set to 2), several dilations, and biases
drawn from quantiles of the convolution output on training windows only. Each (channel, dilation, kernel, bias)
yields one feature: the proportion of positive values (PPV) of the convolution output.
"""
from itertools import combinations

import numpy as np

KERNELS = np.array(list(combinations(range(9), 3)), dtype=np.int64)     # 84 x 3


class MiniRocketLite:
    def __init__(self, dilations=(1, 2, 4, 8, 16), n_bias=2, seed=0, fit_sample=512):
        self.dilations = dilations
        self.n_bias = n_bias
        self.seed = seed
        self.fit_sample = fit_sample

    @staticmethod
    def _conv_all(x, d):
        """x: (n, L) -> (84, n, Lv) convolution outputs at dilation d (valid mode)."""
        L = x.shape[1]
        Lv = L - 8 * d
        S = np.stack([x[:, j * d: j * d + Lv] for j in range(9)], 0)          # (9, n, Lv)
        tot = S.sum(0)
        return -tot[None] + 3.0 * (S[KERNELS[:, 0]] + S[KERNELS[:, 1]] + S[KERNELS[:, 2]])

    def fit(self, X):
        """X: (N, L, C) training windows (no labels are used)."""
        rng = np.random.default_rng(self.seed)
        N, L, C = X.shape
        self.dil_ = [d for d in self.dilations if 8 * d < L - 8]
        idx = rng.choice(N, size=min(self.fit_sample, N), replace=False)
        self.q_ = rng.uniform(0.1, 0.9, size=(C, len(self.dil_), 84, self.n_bias))
        self.bias_ = np.zeros_like(self.q_)
        for c in range(C):
            xc = X[idx, :, c].astype(np.float32)
            for k, d in enumerate(self.dil_):
                out = self._conv_all(xc, d)                                     # (84, n, Lv)
                flat = out.reshape(84, -1)
                for b in range(self.n_bias):
                    self.bias_[c, k, :, b] = [np.quantile(flat[i], self.q_[c, k, i, b]) for i in range(84)]
        return self

    def transform(self, X, chunk=1500):
        N, L, C = X.shape
        F = np.zeros((N, C * len(self.dil_) * 84 * self.n_bias), np.float32)
        for a in range(0, N, chunk):
            xb = X[a:a + chunk].astype(np.float32)
            cols = []
            for c in range(C):
                for k, d in enumerate(self.dil_):
                    out = self._conv_all(xb[:, :, c], d)                         # (84, n, Lv)
                    for b in range(self.n_bias):
                        cols.append((out > self.bias_[c, k, :, b][:, None, None]).mean(-1).T)   # (n, 84)
            F[a:a + chunk] = np.concatenate(cols, 1)
        return F

    @property
    def n_features(self):
        return self.bias_.size


def rocket_channels(X, sensors):
    """Channels fed to MiniRocket: the three axes and the magnitude of every sensor."""
    chans = []
    for ch in sensors.values():
        ax = X[:, :, ch]
        chans += [ax, np.sqrt((ax ** 2).sum(-1, keepdims=True))]
    return np.concatenate(chans, -1).astype(np.float32)
