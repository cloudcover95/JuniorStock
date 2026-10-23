# juniorstock/quant/forecasting/trit_breakout_commit.py
"""Channel breakout trits next to trit_side_commit.py: price vs trailing high/low channel -> balanced trits -> pack -> side_commit. Pure math, stdlib only.

Close above the prior lookback high by more than the band -> +1, below the prior lookback low by more than the band -> -1, otherwise 0.
Per-port commits chain through the five ports in order, each prev being the last commit.
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


def breakout_trits(prices, lookback: int = 3, band: float = 0.0) -> list[int]:
    """One trit per index i >= lookback, comparing prices[i] to the max/min of the prior lookback closes."""
    if lookback < 1:
        raise ValueError("need lookback >= 1")
    out = []
    for i in range(lookback, len(prices)):
        win = prices[i - lookback:i]
        hi, lo, p = max(win), min(win), prices[i]
        up = (p - hi) / hi if hi else 0.0
        dn = (lo - p) / lo if lo else 0.0
        out.append(1 if up > band else (-1 if dn > band else 0))
    return out


def breakout_run(trits) -> int:
    """Signed length of the trailing run of equal non-zero trits (0 if the last trit is 0 or empty)."""
    if not trits or trits[-1] == 0:
        return 0
    k, last = 0, trits[-1]
    for t in reversed(trits):
        if t != last:
            break
        k += 1
    return k * last


def breakout_commit(prices, text: str, prev: str = GENESIS, lookback: int = 3, band: float = 0.0) -> dict:
    t = breakout_trits(prices, lookback, band)
    pk = pack(t)
    return {"prev": prev, "trits": t, "text": text, "pack": pk.hex(), "commit": side_commit(prev, pk, text)}


def port_commits(prices, prev: str = GENESIS, lookback: int = 3, band: float = 0.0) -> dict:
    """Chain one commit per port: text = port name, prev = previous port's commit."""
    pk = pack(breakout_trits(prices, lookback, band))
    out, c = {}, prev
    for port in PORTS:
        c = side_commit(c, pk, port)
        out[port] = c
    return out


def rate_breakout(trits, s0: float = 0.0) -> float:
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
    px = [100.0, 101.0, 100.5, 102.0, 102.0, 99.0, 98.0, 101.5]
    t = breakout_trits(px, lookback=3, band=0.001)
    assert t == [1, 0, -1, -1, 0], t
    assert breakout_trits([1.0, 2.0, 3.0], lookback=3) == []
    assert breakout_trits([5.0, 5.0, 5.0, 5.0], lookback=3, band=0.001) == [0]
    assert breakout_run([0, -1, -1]) == -2 and breakout_run([1, 0]) == 0 and breakout_run([]) == 0
    try:
        breakout_trits(px, lookback=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    d = breakout_commit(px, "AAA", lookback=3, band=0.001)
    assert len(d["commit"]) == 16 and d["commit"] == side_commit(GENESIS, pack(t), "AAA")
    pc = port_commits(px, lookback=3, band=0.001)
    assert tuple(pc) == PORTS and len(set(pc.values())) == 5
    assert pc["Home"] == _side_commit_local(GENESIS, _pack_local(t), "Home")
    assert pc["OSai"] == _side_commit_local(pc["Home"], _pack_local(t), "OSai")
    assert engines_agree(t) and engines_agree([]) and engines_agree([-1, -1, 0, 1, 1, 0, -1])
    assert abs(rate_breakout([1]) - 0.05) < 1e-12
    assert RATE == 0.05 and not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    print({"trits": t, "pack": d["pack"], "commit": d["commit"], "ports": pc, "agree": True, "ok": True})
