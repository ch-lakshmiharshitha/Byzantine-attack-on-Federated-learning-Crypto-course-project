"""Byzantine attacks.

Every attack returns an (f, d) matrix of malicious UPDATES (deltas w_local - w_global).

Threat models
  * data poisoning  : label_flip     (attacker only controls its local data)
  * model poisoning : the rest       (attacker sends arbitrary vectors)
  * omniscient      : alie, ipm, minmax (attacker also knows the benign updates)

ctx (dict) fields:
  benign   : (n-f, d) honest updates of benign clients
  own      : (f, d)   updates the Byzantine clients would send if they were honest
  poisoned : (f, d)   updates trained on label-flipped data (for label_flip)
  n, f, rng
"""
from statistics import NormalDist
import numpy as np


def no_attack(ctx):
    return ctx["own"]


def label_flip(ctx):
    """Data poisoning: clients train honestly but on flipped labels (y -> C-1-y)."""
    return ctx["poisoned"]
def sign_flip(ctx, scale=3.0):
    """Send the negated (and amplified) honest update."""
    return -scale * ctx["own"]

def gaussian(ctx, std=1.0):
    """Random noise update."""
    return ctx["rng"].normal(0.0, std, size=ctx["own"].shape)

def scaling(ctx, boost=10.0):
    """Boosting / model-replacement style: honest-looking direction, huge magnitude, flipped sign
    relative to the benign mean so it dominates FedAvg."""
    mu = ctx["benign"].mean(0)
    return np.tile(-boost * mu, (ctx["f"], 1))

def ipm(ctx, eps=0.5):
    """Inner Product Manipulation (Xie et al. 2020): -eps * mean(benign)."""
    mu = ctx["benign"].mean(0)
    return np.tile(-eps * mu, (ctx["f"], 1))

def alie(ctx):
    """A Little Is Enough (Baruch et al. 2019): mu - z*sigma, with z chosen so that
    the malicious value stays inside the benign distribution (evades Krum/median/trimmed-mean)."""
    n, f = ctx["n"], ctx["f"]
    mu, sd = ctx["benign"].mean(0), ctx["benign"].std(0)
    s = int(np.floor(n / 2 + 1)) - f
    z = NormalDist().inv_cdf(min(max((n - s) / n, 0.5001), 0.9999))
    return np.tile(mu - z * sd, (f, 1))

def minmax(ctx):
    """Min-Max (Shejwalkar & Houmansadr, NDSS'21): perturb along -std, with the largest gamma
    such that the max distance to any benign update <= max pairwise benign distance."""
    B = ctx["benign"]
    mu, direction = B.mean(0), -B.std(0)
    diff = B[:, None, :] - B[None, :, :]
    max_benign = np.sqrt((diff ** 2).sum(-1)).max()
    lo, hi = 0.0, 50.0
    for _ in range(25):
        g = (lo + hi) / 2
        mal = mu + g * direction
        if np.sqrt(((B - mal) ** 2).sum(1)).max() <= max_benign:
            lo = g
        else:
            hi = g
    return np.tile(mu + lo * direction, (ctx["f"], 1))


ATTACKS = {
    "none": no_attack,
    "label_flip": label_flip,
    "sign_flip": sign_flip,
    "gaussian": gaussian,
    "scaling": scaling,
    "ipm": ipm,
    "alie": alie,
    "minmax": minmax,
}
