# juniorstock/quant/forecasting/trit_macd_commit.py
"""MACD histogram trits next to trit_side_commit.py: EMA(fast) - EMA(slow) vs its signal EMA -> balanced trits -> pack -> side_commit. Pure math, stdlib only.

Histogram (MACD - signal) relative to price above +band -> +1, below -band -> -1, otherwise 0.
Per-port commits chain through the five ports in order, each prev being the last commit.
Ports (names only, no addresses): Home, OSai, FieldCore, Flagstaff, JuniorLLM.
"""
from __future__ import annotations

import hashlib
import random

try:
    from .trit_side_commit import LAYER_ONE, MINT, RATE, SOCKET, pack, side_commit, unpack
except ImportError:
    from trit_side_commit import LAYER_ONE, MINT, RATE, SOCKET, pack, side_commit, unpack

FETCH = False
UE5_LAUNCH = False
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
GENESIS = "0" * 16


def ema(xs, span: int) -> list[float]:
    """Standard EMA with alpha = 2 / (span + 1), seeded at the first value."""
    if span < 1:
        raise ValueError("need span >= 1")
    if not xs:
        return []
    a = 2.0 / (span + 1)
    out = [float(xs[0])]
    for x in xs[1:]:
        out.append(a * x + (1.0 - a) * out[-1])
    return out


def macd_hist(prices, fast: int = 3, slow: int = 6, signal: int = 3) -> list[float]:
    """MACD histogram per index: (EMA_fast - EMA_slow) - EMA_signal(MACD)."""
    if not fast < slow:
        raise ValueError("need fast < slow")
    m = [f - s for f, s in zip(ema(prices, fast), ema(prices, slow))]
    return [x - y for x, y in zip(m, ema(m, signal))]


def macd_trits(prices, fast: int = 3, slow: int = 6, signal: int = 3, band: float = 0.0) -> list[int]:
    """One trit per index i >= 1 (index 0 is the seed and always flat), histogram scaled by price."""
    h = macd_hist(prices, fast, slow, signal)
    out = []
    for i in range(1, len(h)):
        p = prices[i]
        r = h[i] / p if p else 0.0
        out.append(1 if r > band else (-1 if r < -band else 0))
    return out


def macd_flips(trits) -> int:
    """Count of sign changes between consecutive non-zero trits."""
    nz = [t for t in trits if t]
    return sum(1 for a, b in zip(nz, nz[1:]) if a != b)


def macd_commit(prices, text: str, prev: str = GENESIS, band: float = 0.0, **kw) -> dict:
    t = macd_trits(prices, band=band, **kw)
    pk = pack(t)
    return {"prev": prev, "trits": t, "text": text, "pack": pk.hex(), "commit": side_commit(prev, pk, text)}


def port_commits(prices, prev: str = GENESIS, band: float = 0.0, **kw) -> dict:
    """Chain one commit per port: text = port name, prev = previous port's commit."""
    pk = pack(macd_trits(prices, band=band, **kw))
    out, c = {}, prev
    for port in PORTS:
        c = side_commit(c, pk, port)
        out[port] = c
    return out


def rate_macd(trits, s0: float = 0.0) -> float:
    """Same accumulator shape as trit_signal_chain.rate_signal: s <- (1 - RATE) * s + RATE * t."""
    s = s0
    for t in trits:
        s = (1.0 - RATE) * s + RATE * t
    return s


def _pack_local(trits) -> bytes:
    """Independent engine: 4-byte big-endian count + 2 bits per trit (t + 1)."""
    n = len(trits)
    bits = 0
    for t in trits:
        bits = (bits << 2) | (t + 1)
    return n.to_bytes(4, "big") + bits.to_bytes((2 * n + 7) // 8 or 1, "big")


def _side_commit_local(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def engines_agree(trits, prev: str = GENESIS, text: str = "juniorstock") -> bool:
    a, b = pack(trits), _pack_local(trits)
    return a == b and unpack(a) == list(trits) and side_commit(prev, a, text) == _side_commit_local(prev, b, text)


if __name__ == "__main__":
    px = [100.0, 101.0, 102.5, 104.0, 103.0, 100.0, 97.5, 96.0, 98.0, 101.0]
    t = macd_trits(px, band=0.0005)
    assert len(t) == len(px) - 1 and t[0] == 1 and -1 in t, t
    assert macd_trits([50.0] * 8, band=0.0005) == [0] * 7
    assert macd_trits([]) == [] and macd_trits([1.0]) == []
    assert macd_flips([1, 0, 1, -1, 0, -1, 1]) == 2 and macd_flips([]) == 0
    for bad in ({"fast": 6, "slow": 3}, {"signal": 0}):
        try:
            macd_trits(px, **bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    d = macd_commit(px, "AAA", band=0.0005)
    assert len(d["commit"]) == 16 and d["commit"] == side_commit(GENESIS, pack(t), "AAA")
    pc = port_commits(px, band=0.0005)
    assert tuple(pc) == PORTS and len(set(pc.values())) == 5
    assert pc["Home"] == _side_commit_local(GENESIS, _pack_local(t), "Home")
    assert pc["OSai"] == _side_commit_local(pc["Home"], _pack_local(t), "OSai")
    rng = random.Random(1028)
    for _ in range(500):
        rt = [rng.choice((-1, 0, 1)) for _ in range(rng.randrange(0, 40))]
        assert engines_agree(rt, text=rng.choice(PORTS))
    assert abs(rate_macd([1]) - 0.05) < 1e-12
    assert RATE == 0.05 and not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    print({"trits": t, "pack": d["pack"], "commit": d["commit"], "ports": pc, "agree": True, "ok": True})
