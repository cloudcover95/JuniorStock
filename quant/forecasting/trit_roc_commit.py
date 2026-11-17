# juniorstock/quant/forecasting/trit_roc_commit.py
"""Rate-of-change band trits next to trit_side_commit.py: ROC over a window -> balanced trits -> pack -> side_commit. Pure math, stdlib only.

ROC = 100 * (close[i] - close[i-window]) / close[i-window]. ROC above up -> +1, below dn -> -1, otherwise 0.
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


def roc(closes, window: int = 5) -> list[float]:
    """ROC percent per index i >= window; zero base -> 0.0."""
    if window < 1:
        raise ValueError("need window >= 1")
    out = []
    for i in range(window, len(closes)):
        base = closes[i - window]
        out.append(100.0 * (closes[i] - base) / base if base else 0.0)
    return out


def roc_trits(closes, window: int = 5, up: float = 2.0, dn: float = -2.0) -> list[int]:
    """One trit per ROC value: +1 above up, -1 below dn, 0 between."""
    if dn > up:
        raise ValueError("need dn <= up")
    return [1 if r > up else (-1 if r < dn else 0) for r in roc(closes, window)]


def roc_flips(trits) -> int:
    """Count of sign changes between consecutive non-zero trits."""
    nz = [t for t in trits if t]
    return sum(1 for a, b in zip(nz, nz[1:]) if a != b)


def roc_commit(closes, text: str, prev: str = GENESIS, **kw) -> dict:
    t = roc_trits(closes, **kw)
    pk = pack(t)
    return {"prev": prev, "trits": t, "text": text, "pack": pk.hex(), "commit": side_commit(prev, pk, text)}


def port_commits(closes, prev: str = GENESIS, **kw) -> dict:
    """Chain one commit per port: text = port name, prev = previous port's commit."""
    pk = pack(roc_trits(closes, **kw))
    out, c = {}, prev
    for port in PORTS:
        c = side_commit(c, pk, port)
        out[port] = c
    return out


def rate_roc(trits, s0: float = 0.0) -> float:
    """s <- (1 - RATE) * s + RATE * t."""
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
    cl = [100.0, 101.0, 102.0, 101.5, 103.0, 110.0, 112.0, 105.0, 99.0, 95.0, 95.5, 96.0]
    t = roc_trits(cl)
    assert len(t) == len(cl) - 5 and 1 in t and -1 in t, t
    assert roc_trits([5.0] * 9) == [0] * 4
    assert roc_trits([]) == [] and roc_trits([1.0, 2.0]) == []
    assert abs(roc([100.0, 110.0], window=1)[0] - 10.0) < 1e-9 and roc([0.0, 1.0], window=1) == [0.0]
    assert roc_flips([1, 0, 1, -1, 0, -1, 1]) == 2 and roc_flips([]) == 0
    for bad in ({"window": 0}, {"up": -5.0}):
        try:
            roc_trits(cl, **bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    d = roc_commit(cl, "AAA")
    assert len(d["commit"]) == 16 and d["commit"] == side_commit(GENESIS, pack(t), "AAA")
    pc = port_commits(cl)
    assert tuple(pc) == PORTS and len(set(pc.values())) == 5
    assert pc["Home"] == _side_commit_local(GENESIS, _pack_local(t), "Home")
    assert pc["OSai"] == _side_commit_local(pc["Home"], _pack_local(t), "OSai")
    rng = random.Random(1117)
    for _ in range(500):
        rt = [rng.choice((-1, 0, 1)) for _ in range(rng.randrange(0, 40))]
        assert engines_agree(rt, text=rng.choice(PORTS))
    assert abs(rate_roc([1]) - 0.05) < 1e-12
    assert RATE == 0.05 and not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    print({"trits": t, "pack": d["pack"], "commit": d["commit"], "ports": pc, "agree": True, "ok": True})
