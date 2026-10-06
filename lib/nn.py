"""
Compact multilayer perceptron in NumPy with explicit back-propagation (no deep-learning framework), used for the
federated and personalised experiments. Gradients are verified by finite differences in tests/test_gradients.py.
"""
import numpy as np


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class MLP:
    """x -> [Linear -> ReLU -> Dropout] x H -> Linear -> softmax (K classes; K=2 for binary tasks)."""

    def __init__(self, d_in, hidden, n_out, seed=0, dropout=0.0):
        rng = np.random.default_rng(seed)
        sizes = [d_in] + list(hidden) + [n_out]
        self.params = {}
        for i in range(len(sizes) - 1):
            self.params[f"W{i}"] = (rng.standard_normal((sizes[i], sizes[i + 1])) * np.sqrt(2.0 / sizes[i])).astype(np.float64)
            self.params[f"b{i}"] = np.zeros(sizes[i + 1])
        self.n_layers = len(sizes) - 1
        self.dropout = dropout
        self.rng = np.random.default_rng(seed + 1)

    # ---------------------------------------------------------------------------------------------
    def forward(self, X, train=False):
        cache = {"a0": X}
        a = X
        for i in range(self.n_layers):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            if i < self.n_layers - 1:
                a = np.maximum(z, 0.0)
                if train and self.dropout > 0:
                    m = (self.rng.random(a.shape) >= self.dropout) / (1.0 - self.dropout)
                    a = a * m
                    cache[f"m{i + 1}"] = m
                cache[f"z{i + 1}"] = z
                cache[f"a{i + 1}"] = a
            else:
                cache["logits"] = z
        return cache

    def predict_proba(self, X, batch=8192):
        out = [softmax(self.forward(X[i:i + batch])["logits"]) for i in range(0, len(X), batch)]
        return np.concatenate(out)

    def loss_grad(self, X, y, w=None, train=True, wd=0.0):
        """Weighted cross-entropy loss and gradients."""
        c = self.forward(X, train=train)
        P = softmax(c["logits"])
        n = len(y)
        w = np.ones(n) if w is None else w
        ws = w.sum()
        loss = -(w * np.log(P[np.arange(n), y] + 1e-12)).sum() / ws
        G = P.copy()
        G[np.arange(n), y] -= 1.0
        G *= (w / ws)[:, None]
        grads = {}
        for i in reversed(range(self.n_layers)):
            a_prev = c[f"a{i}"]
            grads[f"W{i}"] = a_prev.T @ G + wd * self.params[f"W{i}"]
            grads[f"b{i}"] = G.sum(0)
            if i > 0:
                G = G @ self.params[f"W{i}"].T
                if f"m{i}" in c:
                    G = G * c[f"m{i}"]
                G = G * (c[f"z{i}"] > 0)
        loss += 0.5 * wd * sum((self.params[f"W{i}"] ** 2).sum() for i in range(self.n_layers))
        return loss, grads

    def get(self):
        return {k: v.copy() for k, v in self.params.items()}

    def set(self, p):
        self.params = {k: v.copy() for k, v in p.items()}


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, keys=None):
        self.t += 1
        for k in (keys or grads.keys()):
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def class_weights(y, K):
    cnt = np.bincount(y, minlength=K).astype(float)
    w = np.where(cnt > 0, len(y) / (K * np.maximum(cnt, 1)), 0.0)
    return w[y]


def train_local(model, X, y, K, epochs=1, lr=1e-3, batch=256, wd=1e-5, mu=0.0, anchor=None, seed=0,
                keys=None, opt=None):
    """Mini-batch training; optional FedProx proximal term mu/2 ||w - anchor||^2. Returns the optimiser."""
    rng = np.random.default_rng(seed)
    opt = opt or Adam(model.params, lr=lr)
    w_all = class_weights(y, K)
    for _ in range(epochs):
        idx = rng.permutation(len(y))
        for a in range(0, len(y), batch):
            b = idx[a:a + batch]
            _, g = model.loss_grad(X[b], y[b], w_all[b], train=True, wd=wd)
            if mu > 0 and anchor is not None:
                for k in g:
                    g[k] = g[k] + mu * (model.params[k] - anchor[k])
            opt.step(model.params, g, keys=keys)
    return opt
