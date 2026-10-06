"""Federated optimisation of the NumPy MLP: FedAvg / FedProx across people (each person's gateway = one client),
plus on-device personalisation (fine-tuning on the person's own onboarding data)."""
import numpy as np

from lib.nn import MLP, Adam, train_local


def fedavg(clients, d_in, K, rounds=40, local_epochs=1, lr=2e-3, batch=256, hidden=(128, 64), mu=0.0,
           dropout=0.1, seed=0, frac=1.0, val=None, eval_every=5, dp_sigma=0.0, clip=None, select=False):
    """clients: list of (X, y). Returns the global model and the per-round validation history.
    Weighted aggregation by client size; optional FedProx term (mu) and client-update clipping + Gaussian noise.
    `val` is only used to record a learning curve unless select=True (then the best round is kept; use only
    with validation data that is disjoint from the test person)."""
    rng = np.random.default_rng(seed)
    glob = MLP(d_in, hidden, K, seed=seed, dropout=dropout)
    n = np.array([len(c[1]) for c in clients], float)
    hist = []
    best, best_score = glob.get(), -np.inf
    opts = [None] * len(clients)
    for r in range(rounds):
        sel = np.where(rng.random(len(clients)) < frac)[0] if frac < 1 else np.arange(len(clients))
        if len(sel) == 0:
            sel = rng.choice(len(clients), 1)
        g = glob.get()
        upd = {k: np.zeros_like(v) for k, v in g.items()}
        wsum = n[sel].sum()
        for i in sel:
            X, y = clients[i]
            loc = MLP(d_in, hidden, K, seed=seed, dropout=dropout)
            loc.set(g)
            opts[i] = train_local(loc, X, y, K, epochs=local_epochs, lr=lr, batch=batch, mu=mu,
                                  anchor=g if mu > 0 else None, seed=seed * 1000 + r * 37 + int(i), opt=opts[i])
            delta = {k: loc.params[k] - g[k] for k in g}
            if clip is not None:
                norm = np.sqrt(sum((v ** 2).sum() for v in delta.values()))
                f = min(1.0, clip / (norm + 1e-12))
                delta = {k: v * f for k, v in delta.items()}
            for k in g:
                upd[k] += (n[i] / wsum) * delta[k]
        if dp_sigma > 0 and clip is not None:
            for k in upd:
                upd[k] += rng.normal(0, dp_sigma * clip / len(sel), size=upd[k].shape)
        glob.set({k: g[k] + upd[k] for k in g})
        if val is not None and ((r + 1) % eval_every == 0 or r == rounds - 1):
            s = val(glob)
            hist.append((r + 1, s))
            if s > best_score:
                best_score, best = s, glob.get()
    if val is not None and select:
        glob.set(best)
    return glob, hist


def finetune(model, X, y, K, epochs=10, lr=1e-3, batch=64, last_only=True, seed=0):
    """Personalise a copy of `model` on the person's own labelled onboarding data."""
    m = MLP(model.params["W0"].shape[0], [model.params[f"W{i}"].shape[1] for i in range(model.n_layers - 1)], K,
            seed=seed, dropout=0.0)
    m.set(model.get())
    keys = [f"W{m.n_layers - 1}", f"b{m.n_layers - 1}"] if last_only else None
    if len(y) == 0:
        return m
    opt = Adam(m.params, lr=lr)
    for ep in range(epochs):
        train_local(m, X, y, K, epochs=1, lr=lr, batch=batch, seed=seed + ep, keys=keys, opt=opt)
    return m
