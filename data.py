"""Dataset loading and federated (IID / non-IID) partitioning."""
import numpy as np
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split


def load_dataset(name="digits", seed=0, mnist_subset=20000):
    """Return Xtr, ytr, Xte, yte (features standardised to [0,1]-ish floats)."""
    if name == "digits":                       # offline, 1797 samples, 8x8 images
        d = load_digits()
        X, y = d.data / 16.0, d.target
    elif name == "mnist":                      # needs internet on first run
        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X, y = d.data / 255.0, d.target.astype(int)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))[:mnist_subset]
        X, y = X[idx], y[idx]
    else:
        raise ValueError(f"unknown dataset {name}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    return Xtr.astype(np.float64), ytr, Xte.astype(np.float64), yte


def partition(y, n_clients, alpha, rng, min_size=10):
    """Split sample indices among clients.
    alpha=None -> IID. Otherwise Dirichlet(alpha) label skew (smaller = more non-IID)."""
    n = len(y)
    if alpha is None:
        return [a for a in np.array_split(rng.permutation(n), n_clients)]
    classes = np.unique(y)
    while True:
        parts = [[] for _ in range(n_clients)]
        for c in classes:
            idx = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(p) for p in parts]
