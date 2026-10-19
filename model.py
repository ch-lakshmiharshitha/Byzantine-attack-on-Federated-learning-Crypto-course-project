"""Small MLP with manual backprop, parameters stored as ONE flat vector
(so FL updates are plain vectors that attacks/defenses can manipulate)."""
import numpy as np


class MLP:
    def __init__(self, sizes):
        self.sizes = sizes
        self.shapes = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.shapes += [(a, b), (b,)]
        self.dim = int(sum(np.prod(s) for s in self.shapes))

    def init(self, rng):
        parts = []
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            parts.append((rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).ravel())
            parts.append(np.zeros(b))
        return np.concatenate(parts)

    def _unflat(self, w):
        out, i = [], 0
        for s in self.shapes:
            n = int(np.prod(s))
            out.append(w[i:i + n].reshape(s))
            i += n
        return out

    def forward(self, w, X):
        p = self._unflat(w)
        acts, h = [X], X
        L = len(p) // 2
        for l in range(L):
            z = h @ p[2 * l] + p[2 * l + 1]
            h = z if l == L - 1 else np.maximum(z, 0)
            acts.append(h)
        return acts

    @staticmethod
    def _softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def loss_grad(self, w, X, y):
        p = self._unflat(w)
        acts = self.forward(w, X)
        probs = self._softmax(acts[-1])
        m = len(y)
        loss = -np.log(probs[np.arange(m), y] + 1e-12).mean()
        d = probs.copy()
        d[np.arange(m), y] -= 1
        d /= m
        grads = [None] * len(p)
        L = len(p) // 2
        for l in range(L - 1, -1, -1):
            grads[2 * l] = acts[l].T @ d
            grads[2 * l + 1] = d.sum(0)
            if l > 0:
                d = (d @ p[2 * l].T) * (acts[l] > 0)
        return loss, np.concatenate([g.ravel() for g in grads])

    def accuracy(self, w, X, y):
        return float((self.forward(w, X)[-1].argmax(1) == y).mean())

    def predict(self, w, X):
        return self.forward(w, X)[-1].argmax(1)
