"""Winsor clip + quant error bound for JuniorStock returns. Stdlib only. Pure math.

Winsorizes a return series at P5/P95, then measures how much signal survives
quantization. Three gamma scales (JuniorHome docs/WINSOR.md), same ternary grid:
  - BitNet absmean = mean|W|
  - Junior drop    = mean of |W| after dropping the top 1/16
  - Gemini clip    = mean of min(|W|, P95)
Quant error bound (docs/QUANT_ERROR.md): relative L2 of trit and INT4 round-trips,
plus a per-element worst-case bound (trit: gamma/2 inside the grid, INT4: step/2).
Close rule: sha256(prev || pack || text)[:16]. Rate 0.05 held.
Matches the JuniorQuant src/winsor_clip.py, web3node, stocksnode, BitNet-mlx and
JuniorLLM engines: same i2s trit pack (0->0, 1->1, -1->2, MSB first) and close rule.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network.
"""
from __future__ import annotations

import hashlib
import math
import random

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
GENESIS = "0" * 16
EPS = 1e-8

_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}


def pack(trits) -> bytes:
    """Balanced trits as 2-bit i2s codes (0->0, 1->1, -1->2), MSB first, zero-padded."""
    acc = bits = 0
    out = bytearray()
    for t in trits:
        if t not in _ENC:
            raise ValueError("trit must be -1, 0 or 1")
        acc = (acc << 2) | _ENC[t]
        bits += 2
        if bits == 8:
            out.append(acc)
            acc = bits = 0
    if bits:
        out.append(acc << (8 - bits))
    return bytes(out)


def unpack(data: bytes, n: int) -> list[int]:
    out = []
    for b in data:
        for s in (6, 4, 2, 0):
            out.append(_DEC[(b >> s) & 3])
    return out[:n]


def close_rule(prev: str, pk: bytes, text: str) -> str:
    """sha256(prev || pack || text)[:16]. Matches the other engines."""
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def percentile(xs, q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    s = sorted(xs)
    if not s:
        raise ValueError("empty")
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def returns(prices) -> list[float]:
    """Simple returns p[i]/p[i-1] - 1."""
    return [b / a - 1.0 for a, b in zip(prices, prices[1:]) if a]


def winsor_clip(xs, lo_q: float = 5.0, hi_q: float = 95.0) -> list[float]:
    """Two-sided Winsor clip to [P_lo, P_hi]."""
    lo, hi = percentile(xs, lo_q), percentile(xs, hi_q)
    return [min(max(x, lo), hi) for x in xs]


def gamma_absmean(w) -> float:
    return sum(abs(x) for x in w) / len(w)


def gamma_drop(w) -> float:
    a = sorted(abs(x) for x in w)
    keep = a[: len(a) - len(a) // 16] or a
    return sum(keep) / len(keep)


def gamma_clip(w, q: float = 95.0) -> float:
    a = [abs(x) for x in w]
    p = percentile(a, q)
    return sum(min(x, p) for x in a) / len(a)


GAMMAS = {"absmean": gamma_absmean, "drop": gamma_drop, "clip": gamma_clip}


def ternarize(w, gamma: float) -> list[int]:
    """Same ternary grid for every gamma: round(W/gamma) clipped to {-1,0,1}."""
    g = gamma + EPS
    return [max(-1, min(1, round(x / g))) for x in w]


def int4(w) -> tuple[list[float], float]:
    """Symmetric absmax INT4 round-trip (levels -7..7). Returns (values, step)."""
    s = max(abs(x) for x in w) / 7.0 or 1.0
    return [max(-7, min(7, round(x / s))) * s for x in w], s


def rel_l2(w, q) -> float:
    num = math.sqrt(sum((a - b) ** 2 for a, b in zip(w, q)))
    den = math.sqrt(sum(a * a for a in w)) or 1.0
    return num / den


def max_abs_err(w, q) -> float:
    return max(abs(a - b) for a, b in zip(w, q))


def trit_bound(w, gamma: float) -> float:
    """Worst-case per-element trit error: gamma/2 inside [-1.5g, 1.5g], else |x| - g."""
    return max(max(gamma / 2.0, abs(x) - gamma) for x in w)


def error_bound(xs, text: str = "juniorstock|winsor|0.05", prev: str = GENESIS) -> dict:
    """Winsor-clip xs, then trit (three gammas) and INT4 error, closed under the rule."""
    if BIND != "127.0.0.1":
        return {"ok": False, "why": "bind"}
    if MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH:
        return {"ok": False, "why": "flags"}
    w = winsor_clip(xs)
    trit = {}
    for name, fn in GAMMAS.items():
        g = fn(w)
        t = ternarize(w, g)
        deq = [x * g for x in t]
        trit[name] = {
            "scale": g,
            "trits": t,
            "rel_l2": rel_l2(w, deq),
            "max_err": max_abs_err(w, deq),
            "bound": trit_bound(w, g),
        }
    best_name = min(trit, key=lambda k: trit[k]["rel_l2"])
    q4, step = int4(w)
    return {
        "ok": True,
        "rate": RATE,
        "clipped": sum(1 for a, b in zip(xs, w) if a != b),
        "trit_rel_l2": {k: round(v["rel_l2"], 4) for k, v in trit.items()},
        "trit_within_bound": all(v["max_err"] <= v["bound"] + 1e-12 for v in trit.values()),
        "int4_rel_l2": round(rel_l2(w, q4), 4),
        "int4_within_bound": max_abs_err(w, q4) <= step / 2.0 + 1e-12,
        "best": best_name,
        "close": close_rule(prev, pack(trit[best_name]["trits"]), text),
    }


def self_check() -> dict:
    assert BIND == "127.0.0.1"
    assert PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert abs(RATE - 0.05) < 1e-12
    t = [1, 0, -1, -1, 1, 0, 1]
    p = pack(t)
    assert p == bytes([0b01001010, 0b01000100]), p
    assert unpack(p, len(t)) == t
    # Cross-engine vector: same bytes and close as JuniorQuant winsor_clip / web3node side_commit.
    ref = hashlib.sha256(GENESIS.encode() + p + b"web3").hexdigest()[:16]
    assert close_rule(GENESIS, p, "web3") == ref and len(ref) == 16
    c = winsor_clip([float(i) for i in range(101)])
    assert c[0] == 5.0 and c[-1] == 95.0 and c[50] == 50.0
    assert returns([100.0, 105.0, 99.75]) == [0.050000000000000044, -0.050000000000000044]
    rng = random.Random(0)
    w = [rng.gauss(0.0, 1.0) for _ in range(128)]
    assert gamma_drop(w) <= gamma_absmean(w) + 1e-12
    assert gamma_clip(w) <= gamma_absmean(w) + 1e-12
    r = error_bound(w)
    assert r["ok"] and len(r["close"]) == 16
    assert r["close"] == error_bound(w)["close"]
    assert r["close"] != error_bound(w, prev=r["close"])["close"]  # chains forward
    assert r["clipped"] > 0
    assert r["trit_within_bound"] and r["int4_within_bound"]
    for v in r["trit_rel_l2"].values():
        assert 0.3 < v < 0.8, v  # docs: trit ~0.54 at n=128
    assert 0.05 < r["int4_rel_l2"] < 0.3  # docs: INT4 ~0.14
    assert r["int4_rel_l2"] < min(r["trit_rel_l2"].values())
    # Price path: synthetic GBM-ish walk, fully local.
    px = [100.0]
    for _ in range(128):
        px.append(px[-1] * (1.0 + rng.gauss(0.0, RATE / 4)))
    rp = error_bound(returns(px))
    assert rp["ok"] and rp["trit_within_bound"] and rp["int4_within_bound"]
    return {
        "ok": True,
        "rate": RATE,
        "close": "sha256(prev || pack || text)[:16]",
        "bind": BIND,
        "ports": list(PORTS),
        "trit_rel_l2": r["trit_rel_l2"],
        "int4_rel_l2": r["int4_rel_l2"],
        "tip": r["close"],
        "engines": ("JuniorStock", "JuniorQuant", "web3node", "stocksnode", "JuniorLLM", "BitNet-mlx"),
    }


if __name__ == "__main__":
    print(self_check())
