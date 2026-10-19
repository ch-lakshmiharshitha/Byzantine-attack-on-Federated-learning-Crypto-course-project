"""Cryptography component: pairwise-masking Secure Aggregation (simplified Bonawitz et al., CCS'17)
and HMAC-authenticated updates.

Why it matters for this project
  * Secure aggregation hides individual updates: the server only learns SUM(updates).
  * Robust aggregators (Krum, median, trimmed mean, ...) need to inspect INDIVIDUAL updates.
  => privacy (secure aggregation) and Byzantine-robustness are in tension; a malicious client
     can also hide behind its mask. This file demonstrates both facts.
  * HMAC authenticates *who* sent an update, but an authenticated client can still be Byzantine.

Simplifications: no dropout handling / secret sharing; pairwise secrets are generated with
`secrets` instead of a Diffie-Hellman/ECDH key agreement (a real system derives them via ECDH).
"""
import hashlib
import hmac
import secrets
import numpy as np

SCALE = 2 ** 16
MOD = 2 ** 32


def encode(x):                      # float -> fixed-point in Z_{2^32}
    return np.round(x * SCALE).astype(np.int64) % MOD


def decode(v):                      # Z_{2^32} -> float (two's complement)
    v = np.where(v >= MOD // 2, v - MOD, v)
    return v.astype(np.float64) / SCALE


def _prg(seed: bytes, d):           # HMAC-SHA256 counter-mode PRG -> d ints in Z_{2^32}
    out, ctr = bytearray(), 0
    while len(out) < 4 * d:
        out += hmac.new(seed, ctr.to_bytes(8, "big"), hashlib.sha256).digest()
        ctr += 1
    return np.frombuffer(bytes(out[:4 * d]), dtype=">u4").astype(np.int64)


class SecureAggregation:
    def __init__(self, n_clients):
        self.n = n_clients
        self.pair_keys = {(i, j): secrets.token_bytes(32)
                          for i in range(n_clients) for j in range(i + 1, n_clients)}

    def mask(self, i, update):
        y = encode(update)
        for j in range(self.n):
            if j == i:
                continue
            key = self.pair_keys[(min(i, j), max(i, j))]
            m = _prg(key, len(update))
            y = (y + m) % MOD if i < j else (y - m) % MOD
        return y

    def aggregate(self, masked):    # server side: masks cancel in the sum
        return decode(np.sum(masked, axis=0) % MOD)


# ---- HMAC-authenticated update transport --------------------------------------------------
def sign_update(key: bytes, round_no: int, client_id: int, update: np.ndarray) -> bytes:
    msg = round_no.to_bytes(4, "big") + client_id.to_bytes(4, "big") + update.astype(np.float64).tobytes()
    return hmac.new(key, msg, hashlib.sha256).digest()


def verify_update(key, round_no, client_id, update, tag) -> bool:
    return hmac.compare_digest(sign_update(key, round_no, client_id, update), tag)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 8, 1000
    U = rng.normal(0, 0.1, (n, d))
    sa = SecureAggregation(n)
    masked = np.stack([sa.mask(i, U[i]) for i in range(n)])
    print("masked update looks random? corr(update, masked) =",
          round(float(np.corrcoef(U[0], masked[0])[0, 1]), 4))
    print("max |securesum - plainsum| =", float(np.abs(sa.aggregate(masked) - U.sum(0)).max()))

    # A Byzantine client poisons the sum and the server cannot tell which one it was.
    U[3] = -50 * U[3]
    masked = np.stack([sa.mask(i, U[i]) for i in range(n)])
    print("poisoned sum error vs honest mean:", float(np.linalg.norm(sa.aggregate(masked) / n - U[[0,1,2,4,5,6,7]].mean(0))))
    # Authentication
    k = secrets.token_bytes(32)
    tag = sign_update(k, 1, 3, U[3])
    print("HMAC verifies:", verify_update(k, 1, 3, U[3], tag),
          "| tampered:", verify_update(k, 1, 3, U[3] + 1e-6, tag))
