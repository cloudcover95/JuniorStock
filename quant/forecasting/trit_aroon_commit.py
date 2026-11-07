# juniorstock/quant/forecasting/trit_aroon_commit.py
"""Aroon oscillator trits next to trit_side_commit.py: Aroon up/down over a close window -> balanced trits -> pack -> side_commit. Pure math, stdlib only.

Aroon up = 100 * (window - bars since the window high) / window; Aroon down likewise for the low.
Oscillator = up - down. Above +level -> +1, below -level -> -1, otherwise 0.
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


def aroon(prices, window: int = 5) -> list[tuple[float, float]]:
    """(up, down) per index i >= window over the trailing window + 1 closes; ties take the latest bar."""
    if window < 1:
        raise ValueError("need window >= 1")
    out = []
    for i in range(window, len(prices)):
        w = prices[i - window : i + 1]
        hi = max(range(len(w)), key=lambda k: (w[k], k))
        lo = min(range(len(w)), key=lambda k: (w[k], -k))
        out.append((100.0 * hi / window, 100.0 * lo / window))
    return out


def aroon_osc(prices, window: int = 5) -> list[float]:
    return [u - d for u, d in aroon(prices, window)]


def aroon_trits(prices, window: int = 5, level: float = 50.0) -> list[int]:
    """One trit per full window: +1 above +level, -1 below -level, 0 inside the band."""
    if level < 0:
        raise ValueError("need level >= 0")
    return [1 if o > level else (-1 if o < -level else 0) for o in aroon_osc(prices, window)]


def aroon_runs(trits) -> int:
    """Count of maximal runs of equal non-zero trits."""
    runs, last = 0, 0
    for t in trits:
        if t and t != last:
            runs += 1
        last = t
    return runs


def aroon_commit(prices, text: str, prev: str = GENESIS, **kw) -> dict:
    t = aroon_trits(prices, **kw)
    pk = pack(t)
    return {"prev": prev, "trits": t, "text": text, "pack": pk.hex(), "commit": side_commit(prev, pk, text)}


def port_commits(prices, prev: str = GENESIS, **kw) -> dict:
    """Chain one commit per port: text = port name, prev = previous port's commit."""
    pk = pack(aroon_trits(prices, **kw))
    out, c = {}, prev
    for port in PORTS:
        c = side_commit(c, pk, port)
        out[port] = c
    return out


def rate_aroon(trits, s0: float = 0.0) -> float:
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
    px = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0]
    t = aroon_trits(px)
    assert len(t) == len(px) - 5 and 1 in t and -1 in t, t
    assert aroon([1.0, 2.0, 3.0], window=2) == [(100.0, 0.0)]
    assert aroon([3.0, 2.0, 1.0], window=2) == [(0.0, 100.0)]
    assert aroon_trits([50.0] * 8) == [0] * 3
    assert aroon_trits([]) == [] and aroon_trits([1.0, 2.0], window=3) == []
    assert aroon_runs([1, 1, 0, 1, -1, -1, 0, 0, -1]) == 4 and aroon_runs([]) == 0
    for bad in ({"window": 0}, {"level": -1.0}):
        try:
            aroon_trits(px, **bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    d = aroon_commit(px, "AAA")
    assert len(d["commit"]) == 16 and d["commit"] == side_commit(GENESIS, pack(t), "AAA")
    pc = port_commits(px)
    assert tuple(pc) == PORTS and len(set(pc.values())) == 5
    assert pc["Home"] == _side_commit_local(GENESIS, _pack_local(t), "Home")
    assert pc["OSai"] == _side_commit_local(pc["Home"], _pack_local(t), "OSai")
    rng = random.Random(1107)
    for _ in range(500):
        rt = [rng.choice((-1, 0, 1)) for _ in range(rng.randrange(0, 40))]
        assert engines_agree(rt, text=rng.choice(PORTS))
    assert abs(rate_aroon([1]) - 0.05) < 1e-12
    assert RATE == 0.05 and not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    print({"trits": t, "pack": d["pack"], "commit": d["commit"], "ports": pc, "agree": True, "ok": True})
