"""Winsor gamma outlier health: breakdown and collateral trit error under contamination.

Stdlib only. Pure math. No socket, no network, no subprocess.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta" steps 1-3
(`juniorctl health` / `ask` / `night` / `security`, plus tests/test_junior_aie.py and
tests/test_linux_security.py), named by Quant Chief for pair A on 2026-10-16 05:11 MT.
This file is the quant-error and Winsor-bound side of that item: one JSON health line
that a `juniorctl health` or `juniorctl security` step can read from a file or stdout,
saying whether each Winsor gamma survives hostile outliers in a BitNet ternary block.

New capability vs the earlier Winsor files (JuniorQuant src/winsor_clip.py,
src/i2sd_quant_budget.py, JuniorStock risk/winsor_quant_error.py): those measure error on
clean blocks. This one injects k outliers of magnitude M*sigma into an I2_S block
(n=128) and measures, for each gamma in JuniorHome docs/WINSOR.md,
  - gamma inflation  = gamma(contaminated) / gamma(clean)
  - breakdown k*     = smallest k whose inflation exceeds 1 + TOL
  - collateral error = trit rel_l2 of the untouched inlier weights, quantized with the
                       contaminated gamma (how many good weights an attacker zeroes out)
and a security flag: max|W| / gamma_drop above OUTLIER_RATIO marks the block for review.
Three gammas, same ternary grid round(W/gamma) clipped to {-1,0,1}:
  - BitNet absmean = mean|W|                       (breakdown 1/n: one outlier moves it)
  - Junior drop    = mean |W| after dropping top 1/16 (holds while k <= n/16 = 8)
  - Gemini clip    = mean of min(|W|, P95)           (tail fraction 0.05 == RATE)
Clean absmean trit rel_l2 sits near docs/QUANT_ERROR.md (~0.54 at n=128).
Measured at n=128, M=64: breakdown k* is 1 (absmean), 7 (clip, just past the 5% tail)
and 9 (drop, just past n/16). One outlier can even lower absmean rel_l2, since absmean
sits below the error-optimal gamma (~1.22 sigma); the harm shows as zeroed inliers, and
by k=3 (2.3% contamination) absmean collateral passes the budget while drop and clip hold.

Close rule: sha256(prev || pack || text)[:16], pack = i2s trit codes (0->0, 1->1, -1->2,
MSB first). Rate 0.05 held. Same pack and close as the BitNet-mlx, JuniorLLM, web3node,
stocksnode, JuniorQuant and JuniorStock engines.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only (named, never opened). mint/socket/layer_one/fetch/ue5_launch false.
No docker.sock. No model pull, no GGUF download.

Usage: python3 security/winsor_outlier_health.py [--selftest | --health | --security]
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ROADMAP_ITEM = ("JuniorLLM docs/BETA_TO_OS.md Live beta 1-3: juniorctl health/ask/night/"
                "security + tests/test_junior_aie.py, tests/test_linux_security.py")
GENESIS = "0" * 16
BLOCK = 128
DROP = 1.0 / 16.0
TOL = 0.25            # inflation above 1 + TOL counts as broken
OUTLIER_RATIO = 8.0   # max|W| / gamma_drop above this flags the block
ATTACK_M = 64.0       # outlier magnitude in sigma, typical of LLM activation spikes
COLLATERAL_BUDGET = 0.54 + 0.10   # docs/QUANT_ERROR.md trit ~0.54 plus slack
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


def p95(w) -> float:
    a = sorted(abs(x) for x in w)
    idx = max(0, math.ceil((1.0 - RATE) * len(a)) - 1)   # nearest-rank P95
    return a[idx]


def gamma_clip(w) -> float:
    c = p95(w)
    return sum(min(abs(x), c) for x in w) / len(w) + EPS


GAMMAS = (("absmean", gamma_absmean), ("drop", gamma_drop), ("clip", gamma_clip))


# ---- ternary grid and error --------------------------------------------------------

def trit_quant(w, g: float) -> list:
    return [max(-1, min(1, int(math.floor(x / g + 0.5)))) for x in w]


def rel_l2(w, trits, g: float) -> float:
    num = sum((x - g * t) ** 2 for x, t in zip(w, trits))
    den = sum(x * x for x in w) + EPS
    return math.sqrt(num / den)


# ---- contamination -----------------------------------------------------------------

def gauss_block(seed: int, n: int = BLOCK, sigma: float = 1.0) -> list:
    rng = random.Random(seed)
    return [rng.gauss(0.0, sigma) for _ in range(n)]


def contaminate(w, k: int, m: float, seed: int):
    """Replace k positions (deterministic by seed) with +/- m * std(w). Returns (w2, idx)."""
    n = len(w)
    sd = math.sqrt(sum(x * x for x in w) / n) + EPS
    rng = random.Random(seed ^ 0x5EED)
    idx = sorted(rng.sample(range(n), k)) if k else []
    w2 = list(w)
    for j, i in enumerate(idx):
        w2[i] = (m if j % 2 == 0 else -m) * sd
    return w2, idx


def probe(w, k: int, m: float, seed: int) -> dict:
    """Per-gamma inflation and inlier collateral error for one contamination level."""
    w2, idx = contaminate(w, k, m, seed)
    bad = set(idx)
    inl = [x for i, x in enumerate(w2) if i not in bad]
    out = {}
    for name, fn in GAMMAS:
        g0, g1 = fn(w), fn(w2)
        t_in = trit_quant(inl, g1)
        out[name] = {
            "gamma_clean": round(g0, 6),
            "gamma_attacked": round(g1, 6),
            "inflation": round(g1 / g0, 4),
            "collateral_rel_l2": round(rel_l2(inl, t_in, g1), 4),
            "zeroed_inliers": sum(1 for t in t_in if t == 0),
        }
    return out


def breakdown_k(w, m: float = ATTACK_M, seed: int = 0, kmax: int = 24) -> dict:
    """Smallest k whose inflation passes 1 + TOL, per gamma (kmax + 1 if never)."""
    res = {name: kmax + 1 for name, _ in GAMMAS}
    for k in range(1, kmax + 1):
        p = probe(w, k, m, seed)
        for name, _ in GAMMAS:
            if res[name] > kmax and p[name]["inflation"] > 1.0 + TOL:
                res[name] = k
    return res


def security_flag(w) -> dict:
    gd = gamma_drop(w)
    ratio = max(abs(x) for x in w) / gd
    return {"max_over_gamma_drop": round(ratio, 3), "flag": ratio > OUTLIER_RATIO}


# ---- health report (one JSON line, closed under the close rule) --------------------

def health_report(seed: int = 2026, k: int = 3, m: float = ATTACK_M, prev: str = GENESIS) -> dict:
    w = gauss_block(seed)
    w_attacked, _ = contaminate(w, k, m, seed)
    p = probe(w, k, m, seed)
    bk = breakdown_k(w, m, seed)
    verdict = {name: (p[name]["collateral_rel_l2"] <= COLLATERAL_BUDGET) for name, _ in GAMMAS}
    sec_clean, sec_att = security_flag(w), security_flag(w_attacked)
    trits = trit_quant(w_attacked, gamma_drop(w_attacked))
    body = {
        "check": "winsor_outlier_health",
        "roadmap": ROADMAP_ITEM,
        "block": BLOCK, "k": k, "m_sigma": m, "rate": RATE, "tol": TOL,
        "probe": p, "breakdown_k": bk, "within_budget": verdict,
        "security": {"clean": sec_clean, "attacked": sec_att},
        "recommended_gamma": "drop" if verdict["drop"] else ("clip" if verdict["clip"] else "none"),
        "ports": list(PORTS), "bind": BIND,
        "flags": {"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
                  "fetch": FETCH, "ue5_launch": UE5_LAUNCH},
    }
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    body["close"] = close_rule(prev, pack(trits), text)
    body["ok"] = bool((verdict["drop"] or verdict["clip"]) and not sec_clean["flag"]
                      and sec_att["flag"])
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
    v = [float(i) for i in range(1, 17)]            # |W| = 1..16
    assert abs(gamma_absmean(v) - 8.5) < 1e-9
    assert abs(gamma_drop(v) - 8.0) < 1e-9          # drops the single top 1/16 (16)
    assert p95(v) == 16.0 and abs(gamma_clip(v) - 8.5) < 1e-9
    # Clean block: error near docs/QUANT_ERROR.md trit ~0.54, no security flag.
    clean_err = []
    for s in range(32):
        w = gauss_block(1000 + s)
        g = gamma_absmean(w)
        clean_err.append(rel_l2(w, trit_quant(w, g), g))
        assert not security_flag(w)["flag"], s
    mean_err = sum(clean_err) / len(clean_err)
    assert 0.45 < mean_err < 0.60, mean_err
    # One 64-sigma outlier: absmean inflates and zeroes inliers; drop and clip hold.
    w = gauss_block(2026)
    p = probe(w, 1, ATTACK_M, 2026)
    assert p["absmean"]["inflation"] > 1.0 + TOL, p["absmean"]
    assert p["drop"]["inflation"] < 1.0 + TOL and p["clip"]["inflation"] < 1.0 + TOL, p
    assert p["absmean"]["zeroed_inliers"] > p["drop"]["zeroed_inliers"]
    # Three outliers (2.3%): absmean collateral passes the budget; drop and clip hold.
    p3 = probe(w, 3, ATTACK_M, 2026)
    assert p3["absmean"]["collateral_rel_l2"] > COLLATERAL_BUDGET, p3["absmean"]
    assert p3["drop"]["collateral_rel_l2"] <= COLLATERAL_BUDGET, p3["drop"]
    assert p3["clip"]["collateral_rel_l2"] <= COLLATERAL_BUDGET, p3["clip"]
    # Breakdown order: absmean at k=1, clip by the 5% tail, drop holds to n/16.
    bk = breakdown_k(w)
    assert bk["absmean"] == 1, bk
    assert bk["drop"] > int(BLOCK * DROP), bk
    assert bk["clip"] > math.floor(RATE * BLOCK) - 1, bk
    assert bk["absmean"] < bk["clip"] < bk["drop"], bk
    # Health line: deterministic, closed, ok.
    h1, h2 = health_report(), health_report()
    assert h1 == h2 and len(h1["close"]) == 16 and h1["ok"], h1
    assert h1["security"]["attacked"]["flag"] and not h1["security"]["clean"]["flag"]
    # Chain two health lines under the close rule.
    h3 = health_report(seed=2027, prev=h1["close"])
    assert h3["close"] != h1["close"]
    return {"selftest": "pass", "clean_trit_rel_l2": round(mean_err, 4),
            "breakdown_k": bk, "close": h1["close"], "chain": h3["close"]}


def main(argv) -> int:
    if "--selftest" in argv:
        print(json.dumps(self_check(), sort_keys=True))
        return 0
    h = health_report()
    if "--security" in argv:
        h = {"check": h["check"], "security": h["security"], "close": h["close"], "ok": h["ok"]}
    print(json.dumps(h, sort_keys=True))
    return 0 if h["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
