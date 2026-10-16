"""Winsor night drift: certified sampled trit-error bound and night-to-night drift alarm.

Stdlib only. Pure math. No socket, no network, no subprocess.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta" steps 1-3
(`juniorctl health` / `ask` / `night` / `security`, plus tests/test_junior_aie.py and
tests/test_linux_security.py; docs/COMPILED_BACKLOG.md "A - live beta suite"), named again
by Quant Chief for pair A on 2026-10-16 14:15 MT. This file is the quant-error and
Winsor-bound side of the `juniorctl night` step: one JSON line a nightly job can read,
saying with confidence 1 - RATE whether each Winsor gamma's mean trit error stays under
budget, and whether tonight's error drifted from the previous night's.

New capability vs the earlier Winsor files (JuniorQuant src/winsor_clip.py,
src/i2sd_quant_budget.py, src/i2sd_residual_bound.py; JuniorStock
risk/winsor_quant_error.py, security/winsor_outlier_health.py): those measure or bound
error on given blocks. This one samples K blocks (n=128) and turns the sample into a
probability statement, without reading every block of a model:
  - range fact      : round-to-nearest ternary never overshoots, |x - g*t| <= |x|, so
                      per-block trit rel_l2 lies in [0, 1] for any gamma (asserted)
  - UCB             : min(Hoeffding, empirical Bernstein [Maurer-Pontil 2009]) at
                      delta/2 each, so P(true mean > UCB) <= delta = RATE = 0.05
  - sample plan     : smallest K that lets Hoeffding certify the budget, given the gap
  - drift alarm     : two-sample Hoeffding on [0,1] errors, threshold
                      sqrt(ln(1/delta)/2 * (1/K0 + 1/K1)), false alarm <= RATE
Three gammas, same ternary grid round(W/gamma) clipped to {-1,0,1} (JuniorHome docs/WINSOR.md):
  - BitNet absmean = mean|W|
  - Junior drop    = mean |W| after dropping top 1/16
  - P95 clip       = mean of min(|W|, P95)
Clean Gaussian trit rel_l2 sits near docs/QUANT_ERROR.md (~0.54 at n=128).

Close rule: sha256(prev || pack || text)[:16], pack = i2s trit codes (0->0, 1->1, -1->2,
MSB first) of tonight's first block under the drop gamma. Rate 0.05 held. Same pack and
close as the BitNet-mlx, JuniorLLM, web3node, stocksnode, JuniorQuant and JuniorStock engines.
Buddy file (Trit Pack & I2_S): JuniorLLM scripts/live_beta_gate_jtr1.py (7313790).
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only (named, never opened). mint/socket/layer_one/fetch/ue5_launch false.
No docker.sock. No model pull, no GGUF download. Weights come only from an optional local
JSON file (a flat list of floats) or the built-in synthetic blocks.

Usage: python3 quant/calibration/winsor_night_drift.py [--selftest | --night]
       [--weights PATH] [--prev-weights PATH] [--prev HEX16]
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys

RATE = 0.05
DELTA = RATE
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("BitNet-mlx", "JuniorLLM", "web3node", "stocksnode", "JuniorQuant", "JuniorStock")
ROADMAP_ITEM = ("JuniorLLM docs/BETA_TO_OS.md Live beta 1-3: juniorctl health/ask/night/"
                "security + tests/test_junior_aie.py, tests/test_linux_security.py")
GENESIS = "0" * 16
BLOCK = 128
DROP = 1.0 / 16.0
BUDGET = 0.54 + 0.10   # docs/QUANT_ERROR.md trit ~0.54 plus slack
K_NIGHT = 256          # sampled blocks per night
EPS = 1e-12

_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}


# ---- shared engine primitives (pack + close rule) ----------------------------------

def pack(trits) -> bytes:
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


def unpack(data: bytes, n: int) -> list:
    out = []
    for b in data:
        for s in (6, 4, 2, 0):
            out.append(_DEC[(b >> s) & 3])
    return out[:n]


def close_rule(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


# ---- the three Winsor gammas (docs/WINSOR.md) --------------------------------------

def gamma_absmean(w) -> float:
    return sum(abs(x) for x in w) / len(w) + EPS


def gamma_drop(w, drop: float = DROP) -> float:
    a = sorted(abs(x) for x in w)
    keep = len(a) - int(len(a) * drop)
    return sum(a[:keep]) / keep + EPS


def gamma_clip(w) -> float:
    a = sorted(abs(x) for x in w)
    c = a[max(0, math.ceil((1.0 - RATE) * len(a)) - 1)]   # nearest-rank P95
    return sum(min(x, c) for x in a) / len(a) + EPS


GAMMAS = (("absmean", gamma_absmean), ("drop", gamma_drop), ("clip", gamma_clip))


def trit_quant(w, g: float) -> list:
    return [max(-1, min(1, int(math.floor(x / g + 0.5)))) for x in w]


def rel_l2(w, trits, g: float) -> float:
    num = sum((x - g * t) ** 2 for x, t in zip(w, trits))
    den = sum(x * x for x in w) + EPS
    return math.sqrt(num / den)


def block_errors(blocks) -> dict:
    """Per-gamma list of per-block trit rel_l2, each in [0, 1]."""
    out = {name: [] for name, _ in GAMMAS}
    for w in blocks:
        for name, fn in GAMMAS:
            g = fn(w)
            out[name].append(rel_l2(w, trit_quant(w, g), g))
    return out


# ---- concentration bounds on [0, 1] samples ----------------------------------------

def hoeffding_radius(k: int, delta: float) -> float:
    return math.sqrt(math.log(1.0 / delta) / (2.0 * k))


def bernstein_radius(xs, delta: float) -> float:
    """Empirical Bernstein (Maurer & Pontil 2009, Thm 4), one-sided, for [0,1] samples."""
    k = len(xs)
    if k < 2:
        return float("inf")
    m = sum(xs) / k
    var = sum((x - m) ** 2 for x in xs) / (k - 1)
    lg = math.log(2.0 / delta)
    return math.sqrt(2.0 * var * lg / k) + 7.0 * lg / (3.0 * (k - 1))


def ucb(xs, delta: float = DELTA) -> dict:
    """Upper confidence bound on the mean; each bound at delta/2, union gives delta."""
    k = len(xs)
    m = sum(xs) / k
    h = m + hoeffding_radius(k, delta / 2.0)
    b = m + bernstein_radius(xs, delta / 2.0)
    return {"mean": m, "hoeffding": h, "bernstein": b, "ucb": min(h, b, 1.0)}


def sample_plan(mean: float, budget: float = BUDGET, delta: float = DELTA) -> int:
    """Smallest K with mean + hoeffding_radius(K, delta) <= budget (0 if never)."""
    gap = budget - mean
    if gap <= 0:
        return 0
    return math.ceil(math.log(1.0 / delta) / (2.0 * gap * gap))


def drift_threshold(k0: int, k1: int, delta: float = DELTA) -> float:
    return math.sqrt(math.log(1.0 / delta) / 2.0 * (1.0 / k0 + 1.0 / k1))


def drift(prev_xs, xs, delta: float = DELTA) -> dict:
    """One-sided alarm when tonight's mean error rose by more than the Hoeffding threshold."""
    d = sum(xs) / len(xs) - sum(prev_xs) / len(prev_xs)
    t = drift_threshold(len(prev_xs), len(xs), delta)
    return {"rise": d, "threshold": t, "alarm": d > t}


# ---- blocks ------------------------------------------------------------------------

def gauss_blocks(seed: int, k: int = K_NIGHT, n: int = BLOCK, spike: float = 0.0) -> list:
    """k synthetic Gaussian blocks; spike > 0 puts one +/- spike*sigma weight in each."""
    rng = random.Random(seed)
    out = []
    for i in range(k):
        w = [rng.gauss(0.0, 1.0) for _ in range(n)]
        if spike:
            w[rng.randrange(n)] = spike if i % 2 == 0 else -spike
        out.append(w)
    return out


def load_blocks(path: str, n: int = BLOCK, k: int = K_NIGHT, seed: int = 0) -> list:
    """Local JSON flat list of floats, cut into n-blocks, k sampled without replacement."""
    with open(path, "r", encoding="utf-8") as fh:
        flat = [float(x) for x in json.load(fh)]
    blocks = [flat[i:i + n] for i in range(0, len(flat) - n + 1, n)]
    if not blocks:
        raise ValueError("need at least one full block of %d weights" % n)
    if len(blocks) > k:
        blocks = [blocks[i] for i in sorted(random.Random(seed).sample(range(len(blocks)), k))]
    return blocks


# ---- night report (one JSON line, closed under the close rule) ---------------------

def night_report(blocks, prev_blocks=None, prev: str = GENESIS) -> dict:
    errs = block_errors(blocks)
    perr = block_errors(prev_blocks) if prev_blocks else None
    per = {}
    for name, _ in GAMMAS:
        u = ucb(errs[name])
        row = {
            "mean": round(u["mean"], 4),
            "ucb": round(u["ucb"], 4),
            "certified": u["ucb"] <= BUDGET,
            "k_needed_hoeffding": sample_plan(u["mean"]),
        }
        if perr:
            dr = drift(perr[name], errs[name])
            row["drift"] = {"rise": round(dr["rise"], 4),
                            "threshold": round(dr["threshold"], 4), "alarm": dr["alarm"]}
        per[name] = row
    certified = [name for name, _ in GAMMAS if per[name]["certified"]]
    alarms = [name for name, _ in GAMMAS if perr and per[name]["drift"]["alarm"]]
    body = {
        "check": "winsor_night_drift",
        "roadmap": ROADMAP_ITEM,
        "block": BLOCK, "k": len(blocks), "rate": RATE, "delta": DELTA, "budget": BUDGET,
        "gammas": per, "certified": certified, "drift_alarms": alarms,
        "recommended_gamma": next((g for g in ("drop", "clip", "absmean") if g in certified),
                                  "none"),
        "ports": list(PORTS), "bind": BIND, "engines": list(ENGINES),
        "flags": {"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
                  "fetch": FETCH, "ue5_launch": UE5_LAUNCH},
    }
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    first = blocks[0]
    body["close"] = close_rule(prev, pack(trit_quant(first, gamma_drop(first))), text)
    body["ok"] = bool(certified) and not alarms
    return body


# ---- self-test ---------------------------------------------------------------------

def self_check() -> dict:
    assert BIND == "127.0.0.1"
    assert PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert abs(RATE - 0.05) < 1e-12
    # Cross-engine pack + close vector (same bytes as the other engines).
    t = [1, 0, -1, -1, 1, 0, 1]
    pk = pack(t)
    assert pk == bytes([0b01001010, 0b01000100]), pk
    assert unpack(pk, len(t)) == t
    ref = hashlib.sha256(GENESIS.encode() + pk + b"web3").hexdigest()[:16]
    assert close_rule(GENESIS, pk, "web3") == ref and len(ref) == 16
    # Gamma definitions on a tiny exact case.
    v = [float(i) for i in range(1, 17)]
    assert abs(gamma_absmean(v) - 8.5) < 1e-9
    assert abs(gamma_drop(v) - 8.0) < 1e-9
    assert abs(gamma_clip(v) - 8.5) < 1e-9
    # Range fact: no overshoot, per-element |x - g t| <= |x|, so rel_l2 in [0, 1], any gamma.
    rng = random.Random(7)
    for _ in range(200):
        w = [rng.gauss(0, 1) * rng.choice((0.01, 1.0, 50.0)) for _ in range(16)]
        g = rng.uniform(1e-3, 100.0)
        tq = trit_quant(w, g)
        assert all(abs(x - g * q) <= abs(x) + 1e-12 for x, q in zip(w, tq))
        assert 0.0 <= rel_l2(w, tq, g) <= 1.0 + 1e-12
    # Bound algebra.
    assert abs(hoeffding_radius(200, math.exp(-1.0)) - 0.05) < 1e-12
    assert sample_plan(0.54, 0.64, 0.05) == math.ceil(math.log(20.0) / 0.02)
    assert sample_plan(0.70) == 0
    # Clean night: error near docs/QUANT_ERROR.md, certified, UCB above the sample mean.
    night0 = gauss_blocks(2026)
    r0 = night_report(night0)
    for name, _ in GAMMAS:
        g = r0["gammas"][name]
        assert 0.40 < g["mean"] < 0.60, (name, g)
        assert g["mean"] < g["ucb"] <= BUDGET, (name, g)
    assert r0["ok"] and r0["recommended_gamma"] == "drop", r0
    # UCB covers a large-sample reference mean (absmean, 2048 fresh blocks).
    ref_mean = sum(block_errors(gauss_blocks(99, k=2048))["absmean"]) / 2048
    assert ref_mean <= r0["gammas"]["absmean"]["ucb"], ref_mean
    # Same-distribution next night: no drift alarm.
    r1 = night_report(gauss_blocks(2027), night0, prev=r0["close"])
    assert not r1["drift_alarms"] and r1["ok"], r1["gammas"]
    assert r1["close"] != r0["close"] and len(r1["close"]) == 16
    # Spiked night (one 16-sigma weight per block): every gamma alarms and fails budget.
    r2 = night_report(gauss_blocks(2028, spike=16.0), night0, prev=r1["close"])
    assert set(r2["drift_alarms"]) == {"absmean", "drop", "clip"}, r2["gammas"]
    assert not r2["certified"] and not r2["ok"], r2["gammas"]
    # Small K cannot certify: Hoeffding needs k_needed blocks; 8 blocks is too few.
    r3 = night_report(gauss_blocks(5, k=8))
    assert not r3["certified"], r3["gammas"]
    assert r3["gammas"]["drop"]["k_needed_hoeffding"] > 8
    # Deterministic.
    assert night_report(night0) == r0
    return {"selftest": "pass", "close": r0["close"], "chain": r1["close"],
            "ucb": {n: r0["gammas"][n]["ucb"] for n, _ in GAMMAS},
            "spike_rise": {n: r2["gammas"][n]["drift"]["rise"] for n, _ in GAMMAS},
            "drift_threshold": r1["gammas"]["absmean"]["drift"]["threshold"]}


def _arg(argv, flag):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def main(argv) -> int:
    if "--selftest" in argv:
        print(json.dumps(self_check(), sort_keys=True))
        return 0
    wp, pp = _arg(argv, "--weights"), _arg(argv, "--prev-weights")
    prev = _arg(argv, "--prev") or GENESIS
    blocks = load_blocks(wp) if wp else gauss_blocks(2026)
    prev_blocks = load_blocks(pp) if pp else None
    r = night_report(blocks, prev_blocks, prev)
    print(json.dumps(r, sort_keys=True))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
