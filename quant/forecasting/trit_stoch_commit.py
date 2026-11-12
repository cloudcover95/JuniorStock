# juniorstock/quant/forecasting/trit_stoch_commit.py
"""Stochastic %K trits next to trit_side_commit.py: %K over a high/low/close window -> balanced trits -> pack -> side_commit. Pure math, stdlib only.

%K = 100 * (close - lowest low) / (highest high - lowest low). %K above hi -> +1, below lo -> -1, otherwise 0.
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


def stoch_k(highs, lows, closes, window: int = 5) -> list[float]:
    """%K per index i >= window - 1; flat range -> 50.0."""
    if window < 2:
        raise ValueError("need window >= 2")
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("need equal-length highs, lows, closes")
    out = []
    for i in range(window - 1, len(closes)):
        hh = max(highs[i - window + 1 : i + 1])
        ll = min(lows[i - window + 1 : i + 1])
        out.append(100.0 * (closes[i] - ll) / (hh - ll) if hh > ll else 50.0)
    return out


def stoch_trits(highs, lows, closes, window: int = 5, lo: float = 20.0, hi: float = 80.0) -> list[int]:
    """One trit per full window: +1 above hi, -1 below lo, 0 between."""
    if lo > hi:
        raise ValueError("need lo <= hi")
    return [1 if k > hi else (-1 if k < lo else 0) for k in stoch_k(highs, lows, closes, window)]


def stoch_flips(trits) -> int:
    """Count of sign changes between consecutive non-zero trits."""
    nz = [t for t in trits if t]
    return sum(1 for a, b in zip(nz, nz[1:]) if a != b)


def stoch_commit(highs, lows, closes, text: str, prev: str = GENESIS, **kw) -> dict:
    t = stoch_trits(highs, lows, closes, **kw)
    pk = pack(t)
    return {"prev": prev, "trits": t, "text": text, "pack": pk.hex(), "commit": side_commit(prev, pk, text)}


def port_commits(highs, lows, closes, prev: str = GENESIS, **kw) -> dict:
    """Chain one commit per port: text = port name, prev = previous port's commit."""
    pk = pack(stoch_trits(highs, lows, closes, **kw))
    out, c = {}, prev
    for port in PORTS:
        c = side_commit(c, pk, port)
        out[port] = c
    return out


def rate_stoch(trits, s0: float = 0.0) -> float:
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
    cl = [10.0, 10.5, 10.2, 10.4, 10.1, 14.0, 16.0, 15.5, 11.0, 7.0, 6.0, 6.5]
    hi_ = [c + 0.5 for c in cl]
    lo_ = [c - 0.5 for c in cl]
    t = stoch_trits(hi_, lo_, cl)
    assert len(t) == len(cl) - 4 and 1 in t and -1 in t, t
    assert stoch_trits([5.0] * 8, [5.0] * 8, [5.0] * 8) == [0] * 4
    assert stoch_trits([], [], []) == [] and stoch_trits([1.0], [0.0], [0.5], window=3) == []
    assert abs(stoch_k([3.0, 3.0, 3.0], [1.0, 1.0, 1.0], [1.0, 2.0, 3.0], window=3)[0] - 100.0) < 1e-9
    assert stoch_flips([1, 0, 1, -1, 0, -1, 1]) == 2 and stoch_flips([]) == 0
    for bad in ({"window": 1}, {"lo": 90.0}):
        try:
            stoch_trits(hi_, lo_, cl, **bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    try:
        stoch_k([1.0], [1.0, 2.0], [1.0])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    d = stoch_commit(hi_, lo_, cl, "AAA")
    assert len(d["commit"]) == 16 and d["commit"] == side_commit(GENESIS, pack(t), "AAA")
    pc = port_commits(hi_, lo_, cl)
    assert tuple(pc) == PORTS and len(set(pc.values())) == 5
    assert pc["Home"] == _side_commit_local(GENESIS, _pack_local(t), "Home")
    assert pc["OSai"] == _side_commit_local(pc["Home"], _pack_local(t), "OSai")
    rng = random.Random(1112)
    for _ in range(500):
        rt = [rng.choice((-1, 0, 1)) for _ in range(rng.randrange(0, 40))]
        assert engines_agree(rt, text=rng.choice(PORTS))
    assert abs(rate_stoch([1]) - 0.05) < 1e-12
    assert RATE == 0.05 and not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    print({"trits": t, "pack": d["pack"], "commit": d["commit"], "ports": pc, "agree": True, "ok": True})
