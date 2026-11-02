# juniorstock/quant/forecasting/trit_cci_commit.py
"""Commodity Channel Index trits next to trit_side_commit.py: CCI over a close window -> balanced trits -> pack -> side_commit. Pure math, stdlib only.

CCI = (p - SMA) / (0.015 * mean absolute deviation). CCI above +level -> +1, below -level -> -1, otherwise 0.
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
LAMBERT = 0.015


def cci(prices, window: int = 5) -> list[float]:
    """CCI per index i >= window - 1 over the trailing window of closes; flat window -> 0.0."""
    if window < 2:
        raise ValueError("need window >= 2")
    out = []
    for i in range(window - 1, len(prices)):
        w = prices[i - window + 1 : i + 1]
        m = sum(w) / window
        mad = sum(abs(x - m) for x in w) / window
        out.append((prices[i] - m) / (LAMBERT * mad) if mad else 0.0)
    return out


def cci_trits(prices, window: int = 5, level: float = 100.0) -> list[int]:
    """One trit per full window: +1 above +level, -1 below -level, 0 inside the channel."""
    if level < 0:
        raise ValueError("need level >= 0")
    return [1 if c > level else (-1 if c < -level else 0) for c in cci(prices, window)]


def cci_flips(trits) -> int:
    """Count of sign changes between consecutive non-zero trits."""
    nz = [t for t in trits if t]
    return sum(1 for a, b in zip(nz, nz[1:]) if a != b)


def cci_commit(prices, text: str, prev: str = GENESIS, **kw) -> dict:
    t = cci_trits(prices, **kw)
    pk = pack(t)
    return {"prev": prev, "trits": t, "text": text, "pack": pk.hex(), "commit": side_commit(prev, pk, text)}


def port_commits(prices, prev: str = GENESIS, **kw) -> dict:
    """Chain one commit per port: text = port name, prev = previous port's commit."""
    pk = pack(cci_trits(prices, **kw))
    out, c = {}, prev
    for port in PORTS:
        c = side_commit(c, pk, port)
        out[port] = c
    return out


def rate_cci(trits, s0: float = 0.0) -> float:
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
    px = [100.0, 100.5, 100.2, 100.4, 100.1, 104.0, 106.0, 105.5, 101.0, 97.0, 96.0, 96.5]
    t = cci_trits(px)
    assert len(t) == len(px) - 4 and 1 in t and -1 in t, t
    assert cci_trits([50.0] * 8) == [0] * 4
    assert cci_trits([]) == [] and cci_trits([1.0, 2.0], window=3) == []
    assert abs(cci([1.0, 2.0, 3.0], window=3)[0] - 100.0) < 1e-9
    assert cci_flips([1, 0, 1, -1, 0, -1, 1]) == 2 and cci_flips([]) == 0
    for bad in ({"window": 1}, {"level": -1.0}):
        try:
            cci_trits(px, **bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    d = cci_commit(px, "AAA")
    assert len(d["commit"]) == 16 and d["commit"] == side_commit(GENESIS, pack(t), "AAA")
    pc = port_commits(px)
    assert tuple(pc) == PORTS and len(set(pc.values())) == 5
    assert pc["Home"] == _side_commit_local(GENESIS, _pack_local(t), "Home")
    assert pc["OSai"] == _side_commit_local(pc["Home"], _pack_local(t), "OSai")
    rng = random.Random(1102)
    for _ in range(500):
        rt = [rng.choice((-1, 0, 1)) for _ in range(rng.randrange(0, 40))]
        assert engines_agree(rt, text=rng.choice(PORTS))
    assert abs(rate_cci([1]) - 0.05) < 1e-12
    assert RATE == 0.05 and not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    print({"trits": t, "pack": d["pack"], "commit": d["commit"], "ports": pc, "agree": True, "ok": True})
