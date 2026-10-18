# juniorstock/quant/forecasting/trit_crossover_commit.py
"""Moving-average crossover trits next to trit_side_commit.py: fast vs slow mean -> balanced trits -> pack -> side_commit. Pure math, stdlib only.

Fast mean above slow by more than the band -> +1, below by more than the band -> -1, otherwise 0.
Ports (names only, no addresses): Home, OSai, FieldCore, Flagstaff, JuniorLLM.
"""
from __future__ import annotations

import hashlib

try:
    from .trit_side_commit import LAYER_ONE, MINT, RATE, SOCKET, pack, side_commit, unpack
except ImportError:
    from trit_side_commit import LAYER_ONE, MINT, RATE, SOCKET, pack, side_commit, unpack

FETCH = False
UE5_LAUNCH = False
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
GENESIS = "0" * 16


def _mean(xs) -> float:
    return sum(xs) / len(xs)


def crossover_trits(prices, fast: int = 2, slow: int = 4, band: float = 0.0) -> list[int]:
    """One trit per index i >= slow - 1, comparing trailing fast and slow means (relative gap vs band)."""
    if fast < 1 or slow <= fast:
        raise ValueError("need 1 <= fast < slow")
    out = []
    for i in range(slow - 1, len(prices)):
        f = _mean(prices[i - fast + 1:i + 1])
        s = _mean(prices[i - slow + 1:i + 1])
        g = (f - s) / s if s else 0.0
        out.append(1 if g > band else (-1 if g < -band else 0))
    return out


def crossover_events(trits) -> list[int]:
    """Indices where the trit changes sign through or out of zero (a regime flip)."""
    return [i for i in range(1, len(trits)) if trits[i] != trits[i - 1]]


def crossover_commit(prices, text: str, prev: str = GENESIS, fast: int = 2, slow: int = 4, band: float = 0.0) -> dict:
    t = crossover_trits(prices, fast, slow, band)
    pk = pack(t)
    return {"prev": prev, "trits": t, "text": text, "pack": pk.hex(), "commit": side_commit(prev, pk, text)}


def rate_crossover(trits, s0: float = 0.0) -> float:
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
    px = [100.0, 101.0, 102.0, 103.0, 101.0, 99.0, 98.0, 98.0]
    t = crossover_trits(px, fast=2, slow=4, band=0.001)
    assert t == [1, 1, -1, -1, -1], t
    assert crossover_trits([1.0, 2.0, 3.0]) == []
    assert crossover_trits([5.0, 5.0, 5.0, 5.0], band=0.001) == [0]
    assert crossover_events([1, 1, 0, -1, -1, 1]) == [2, 3, 5]
    try:
        crossover_trits(px, fast=4, slow=4)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    d = crossover_commit(px, "AAA", fast=2, slow=4, band=0.001)
    assert len(d["commit"]) == 16 and d["commit"] == side_commit(GENESIS, pack(t), "AAA")
    assert engines_agree(t) and engines_agree([]) and engines_agree([-1, -1, 0, 1, 1, 0, -1])
    assert abs(rate_crossover([1]) - 0.05) < 1e-12
    assert RATE == 0.05 and not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
    print({"trits": t, "pack": d["pack"], "commit": d["commit"], "agree": True, "ok": True})
