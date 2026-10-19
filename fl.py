"""Federated learning simulator with Byzantine clients."""
import numpy as np
from attacks import ATTACKS
from defenses import DEFENSES
from data import partition
from model import MLP


def local_train(model, w_global, X, y, epochs, lr, bs, rng):
    w = w_global.copy()
    n = len(y)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            _, g = model.loss_grad(w, X[b], y[b])
            w -= lr * g
    return w - w_global


def run_fl(Xtr, ytr, Xte, yte, *, attack="none", defense="fedavg", n_clients=20,
           byz_frac=0.2, rounds=30, local_epochs=2, lr=0.1, bs=32, alpha=None,
           hidden=64, seed=0, root_size=100, verbose=False):
    rng = np.random.default_rng(seed)
    n_classes = int(ytr.max()) + 1
    model = MLP([Xtr.shape[1], hidden, n_classes])
    w = model.init(rng)

    # FLTrust root data is held out from clients
    root_idx = rng.choice(len(ytr), root_size, replace=False)
    mask = np.ones(len(ytr), bool)
    mask[root_idx] = False
    Xc, yc = Xtr[mask], ytr[mask]
    parts = partition(yc, n_clients, alpha, rng)

    f = int(round(byz_frac * n_clients)) if attack != "none" else 0
    byz = set(rng.choice(n_clients, f, replace=False).tolist()) if f else set()
    benign_ids = [i for i in range(n_clients) if i not in byz]
    byz_ids = sorted(byz)
    agg_fn, atk_fn = DEFENSES[defense], ATTACKS[attack]
    f_assumed = f if f else int(round(byz_frac * n_clients))  # defense knows the bound
    if attack == "none":
        f_assumed = 0 if defense == "fedavg" else int(round(byz_frac * n_clients))

    hist = {"acc": [], "loss": [], "byz_selected": []}
    for r in range(rounds):
        crng = np.random.default_rng(seed * 100003 + r)

        def upd(i, labels=None):
            idx = parts[i]
            yy = yc[idx] if labels is None else labels
            return local_train(model, w, Xc[idx], yy, local_epochs, lr, bs, crng)

        B = np.stack([upd(i) for i in benign_ids])
        if f:
            own = np.stack([upd(i) for i in byz_ids])
            pois = np.stack([upd(i, n_classes - 1 - yc[parts[i]]) for i in byz_ids]) \
                if attack == "label_flip" else own
            ctx = dict(benign=B, own=own, poisoned=pois, n=n_clients, f=f, rng=crng)
            M = atk_fn(ctx)
            U = np.vstack([B, M])                       # malicious rows are the LAST f rows
            is_byz = np.array([False] * len(B) + [True] * f)
        else:
            U, is_byz = B, np.zeros(len(B), bool)

        kw = {}
        if defense == "fltrust":
            ridx = root_idx
            kw["server_update"] = local_train(model, w, Xtr[ridx], ytr[ridx], local_epochs, lr, bs, crng)
        agg, sel = agg_fn(U, f=f_assumed, **kw)
        w = w + agg

        acc = model.accuracy(w, Xte, yte)
        loss, _ = model.loss_grad(w, Xte, yte)
        hist["acc"].append(acc)
        hist["loss"].append(float(loss) if np.isfinite(loss) else float("nan"))
        if sel is not None and f:
            hist["byz_selected"].append(float(np.mean(is_byz[sel])) if len(sel) else 0.0)
        if verbose:
            print(f"round {r+1:3d} acc={acc:.4f} loss={loss:.4f}")
    return hist
