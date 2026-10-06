"""Finite-difference check of the NumPy MLP back-propagation (run: python tests/test_gradients.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.nn import MLP  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 7))
    y = rng.integers(0, 3, 20)
    w = rng.random(20) + 0.5
    m = MLP(7, (6, 5), 3, seed=1, dropout=0.0)
    # small non-zero biases avoid evaluating ReLU exactly at its kink (z = 0), where the derivative is undefined
    for k in m.params:
        if k.startswith("b"):
            m.params[k] = rng.normal(0, 0.1, m.params[k].shape)
    _, g = m.loss_grad(X, y, w, train=False, wd=1e-3)
    errs = []
    for k, P in m.params.items():
        for idx in [tuple(rng.integers(0, s) for s in P.shape) for _ in range(6)]:
            old = P[idx]
            P[idx] = old + 1e-6; lp, _ = m.loss_grad(X, y, w, train=False, wd=1e-3)
            P[idx] = old - 1e-6; lm, _ = m.loss_grad(X, y, w, train=False, wd=1e-3)
            P[idx] = old
            num = (lp - lm) / 2e-6
            errs.append(abs(num - g[k][idx]) / max(1e-8, abs(num) + abs(g[k][idx])))
    print("relative gradient error: median %.2e, max %.2e" % (np.median(errs), np.max(errs)))
    assert np.max(errs) < 1e-4, "gradient check failed"
    print("OK")


if __name__ == "__main__":
    main()
