"""Aggregation rules. Each takes U (n, d) client updates and f (assumed # Byzantine)
and returns (aggregate (d,), selected_indices or None)."""
import numpy as np


def fedavg(U, f=0, weights=None, **kw):
    if weights is None:
        return U.mean(0), None
    w = np.asarray(weights, float)
    return (w[:, None] * U).sum(0) / w.sum(), None


def coord_median(U, f=0, **kw):
    return np.median(U, axis=0), None


def trimmed_mean(U, f=0, **kw):
    n = len(U)
    b = min(f, (n - 1) // 2)
    S = np.sort(U, axis=0)
    return S[b:n - b].mean(0), None


def _pairwise_sq(U):
    sq = (U ** 2).sum(1)
    D = sq[:, None] + sq[None, :] - 2 * U @ U.T
    np.maximum(D, 0, out=D)
    np.fill_diagonal(D, np.inf)
    return D


def _krum_scores(D, f):
    n = len(D)
    k = max(n - f - 2, 1)
    return np.sort(D, axis=1)[:, :k].sum(1)


def krum(U, f=0, **kw):
    """Blanchard et al. 2017: pick the single update closest to its n-f-2 neighbours."""
    i = int(np.argmin(_krum_scores(_pairwise_sq(U), f)))
    return U[i].copy(), [i]


def multi_krum(U, f=0, m=None, **kw):
    """Average of the m updates with the best Krum scores (default m = n - f)."""
    n = len(U)
    m = m or max(n - f, 1)
    idx = np.argsort(_krum_scores(_pairwise_sq(U), f))[:m]
    return U[idx].mean(0), idx.tolist()


def bulyan(U, f=0, **kw):
    """El Mhamdi et al. 2018: iterative Krum selection of n-2f updates, then
    coordinate-wise trimmed mean around the median. Needs n >= 4f+3 (falls back gracefully)."""
    n = len(U)
    f = min(f, max((n - 3) // 4, 0))
    theta = max(n - 2 * f, 1)
    remaining, chosen = list(range(n)), []
    while len(chosen) < theta and remaining:
        sub = U[remaining]
        if len(sub) > 2:
            j = int(np.argmin(_krum_scores(_pairwise_sq(sub), f)))
        else:
            j = 0
        chosen.append(remaining.pop(j))
    S = U[chosen]
    beta = max(theta - 2 * f, 1)
    med = np.median(S, axis=0)
    order = np.argsort(np.abs(S - med), axis=0)[:beta]
    agg = np.take_along_axis(S, order, axis=0).mean(0)
    return agg, chosen


def geometric_median(U, f=0, iters=100, eps=1e-8, **kw):
    """Weiszfeld's algorithm (RFA, Pillutla et al.)."""
    z = U.mean(0)
    for _ in range(iters):
        d = np.maximum(np.linalg.norm(U - z, axis=1), eps)
        w = 1.0 / d
        z_new = (w[:, None] * U).sum(0) / w.sum()
        if np.linalg.norm(z_new - z) < eps:
            z = z_new
            break
        z = z_new
    return z, None


def norm_clip(U, f=0, **kw):
    """Clip every update to the median norm, then average (Sun et al. 2019 style)."""
    norms = np.linalg.norm(U, axis=1)
    tau = np.median(norms)
    scale = np.minimum(1.0, tau / np.maximum(norms, 1e-12))
    return (U * scale[:, None]).mean(0), None


def fltrust(U, f=0, server_update=None, **kw):
    """Cao et al. 2021: server holds a small clean root dataset; weight clients by
    ReLU(cosine similarity to the server update) and rescale to the server update's norm."""
    g0 = server_update
    n0 = np.linalg.norm(g0)
    cos = (U @ g0) / (np.linalg.norm(U, axis=1) * n0 + 1e-12)
    ts = np.maximum(cos, 0.0)
    if ts.sum() < 1e-12:
        return np.zeros_like(g0), []
    Un = U * (n0 / np.maximum(np.linalg.norm(U, axis=1), 1e-12))[:, None]
    return (ts[:, None] * Un).sum(0) / ts.sum(), np.where(ts > 0)[0].tolist()


DEFENSES = {
    "fedavg": fedavg,
    "median": coord_median,
    "trimmed_mean": trimmed_mean,
    "krum": krum,
    "multi_krum": multi_krum,
    "bulyan": bulyan,
    "geomed": geometric_median,
    "norm_clip": norm_clip,
    "fltrust": fltrust,
}
