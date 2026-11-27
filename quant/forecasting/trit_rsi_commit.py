# juniorstock/quant/forecasting/trit_rsi_commit.py
"""Wilder RSI band trits next to trit_side_commit.py: RSI over closes -> balanced trits -> pack -> side_commit. Pure math, stdlib only.

RSI above hi -> +1, below lo -> -1, otherwise 0. Per-port commits chain through the five ports in order.
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


def rsi(closes, period: int = 14) -> list[float]:
    """Wilder RSI, one value per index >= period; flat window -> 50.0."""
    if period < 2:
        raise ValueError("need period >= 2")
    if len(closes) <= period:
        return []
    d = [b - a for a, b in zip(closes, closes[1:])]
    ag = sum(max(x, 0.0) for x in d[:period]) / period
    al = sum(max(-x, 0.0) for x in d[:period]) / period

    def val(g, l):
        if g == 0 and l == 0:
            return 50.0
        return 100.0 if l == 0 else 100.0 - 100.0 / (1.0 + g / l)

    out = [val(ag, al)]
    for x in d[period:]:
        ag = (ag * (period - 1) + max(x, 0.0)) / period
        al = (al * (period - 1) + max(-x, 0.0)) / period
        out.append(val(ag, al))
    return out


def rsi_trits(closes, period: int = 14, lo: float = 30.0, hi: float = 70.0) -> list[int]:
    if lo > hi:
        raise ValueError("need lo <= hi")
    return [1 if v > hi else (-1 if v < lo else 0) for v in rsi(closes, period)]


def rsi_commit(closes, text: str, prev: str = GENESIS, **kw) -> dict:
    t = rsi_trits(closes, **kw)
    pk = pack(t)
    return {"prev": prev, "trits": t, "text": text, "pack": pk.hex(), "commit": side_commit(prev, pk, text)}


def port_commits(closes, prev: str = GENESIS, **kw) -> dict:
    pk = pack(rsi_trits(closes, **kw))
    out, c = {}, prev
    for port in PORTS:
        c = side_commit(c, pk, port)
        out[port] = c
    return out


def rate_rsi(trits, s0: float = 0.0) -> float:
    s = s0
    for t in trits:
        s = (1.0 - RATE) * s + RATE * t
    return s


def _pack_local(trits) -> bytes:
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
    up = [float(i) for i in range(1, 10)]
    assert rsi(up, 3)[0] == 100.0 and rsi_trits(up, 3) == [1] * 6
    dn = [float(i) for i in range(10, 1, -1)]
    assert rsi_trits(dn, 3) == [-1] * 6
    assert rsi_trits([5.0] * 8, 3) == [0] * 5 and rsi([1.0, 2.0], 3) == [] and rsi_trits([], 3) == []
    zz = [10, 11, 10, 11, 10, 11, 10, 11, 10]
    assert all(0 < v < 100 for v in rsi([float(x) for x in zz], 3))
    for bad in ({"period": 1}, {"lo": 80.0}):
        try:
            rsi_trits(up, **bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    t = rsi_trits(up + dn, 3)
    d = rsi_commit(up + dn, "AAA", period=3)
    assert len(d["commit"]) == 16 and d["commit"] == side_commit(GENESIS, pack(t), "AAA")
    pc = port_commits(up + dn, period=3)
    assert tuple(pc) == PORTS and len(set(pc.values())) == 5
    assert pc["Home"] == _side_commit_local(GENESIS, _pack_local(t), "Home")
    assert pc["OSai"] == _side_commit_local(pc["Home"], _pack_local(t), "OSai")
    rng = random.Random(1127)
    for _ in range(500):
        rt = [rng.choice((-1, 0, 1)) for _ in range(rng.randrange(0, 40))]
        assert engines_agree(rt, text=rng.choice(PORTS))
    assert abs(rate_rsi([1]) - 0.05) < 1e-12
    assert RATE == 0.05 and not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    print({"trits": t, "commit": d["commit"], "agree": True, "ok": True})
