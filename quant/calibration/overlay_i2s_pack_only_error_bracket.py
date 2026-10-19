"""JuniorOS overlay: pack-only Winsor quant-error bracket for an on-disk I2_S pack.

Stdlib only, offline, read-only. Ports touched: Home, OSai, FieldCore,
Flagstaff, JuniorLLM. TCP ports named, never dialed: bitnetd 127.0.0.1:8765,
i2sd userspace 127.0.0.1:8767. Bind 127.0.0.1 only; this file opens no socket.
mint false, socket false, layer_one false, fetch false, ue5_launch false.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay", the clause
"I2_S 2B4T only if GGUF already on disk" (Quant Chief, pair A, slot
2026-10-19 05:20 MT, "JuniorOS overlay, remaining gaps only"). This is the
quant-error / Winsor-bound side. PATH, Debian, os-release.junior, the hardened
bitnetd.service MemoryMax and user-unit parity already landed; the "No kernel
blob / no OSWorld" receipts belong to other teams; the I2_S key side belongs
to Trit Pack & I2_S. This file never opens or downloads a GGUF, never pulls a
model, and writes no OS config.

The gap: on the overlay host the only thing on disk is the ternary pack plus
its stored scale s. The full-precision weights W are gone, so every earlier
quant-error bound (which needs W) cannot run there. Nothing says what the pack
alone certifies about the quantization error, or whether the stored scale is
even consistent with the Winsor gamma the build claims.

New capability: a certified error bracket and variant-consistency trit from
the pack alone (trit counts n0 zeros, n1 nonzeros, n = n0 + n1), the stored
scale s, an optional BitNet eps (gamma = s - eps) and an optional declared
amax A >= max|W|. Code rule: trit t != 0 iff |W| > theta = s / 2, so
  zero    |W| in [0, theta]           nonzero |W| in (theta, A]
and the error is e = |W| - s|t|. For the three Winsor gammas
(JuniorHome docs/WINSOR.md) the sum S = sum |W| satisfies, exactly:
  absmean  gamma = mean|W|                      S = n gamma
  drop     gamma = mean |W| of the smallest m = n - floor(n/16)
           (each dropped |W| >= max kept >= gamma)
                                                S in [n gamma, n gamma + (n-m)(A-gamma)]
  clip     gamma = mean min(|W|, P95), nearest rank r = ceil(0.95 n),
           P95 >= gamma, at most q = n - r above P95
                                                S in [n gamma, n gamma + q (A-gamma)]
Intersecting with the pack's own range S in [n1 theta, n0 theta + n1 A]:
  consistent  +1 nonempty at gamma; 0 empty at gamma but nonempty with the
              gamma slack RATE (gamma * (1 -+ 0.05), f16/f32 scale rounding and
              blockwise stats); -1 empty even with slack (the stored scale is
              not that Winsor gamma of any weights with this pack).
  amax_floor  max|W| >= max(theta, (n gamma - n0 theta) / n1): a certified
              lower bound on the largest weight, from the pack alone.
  sse_min     exact water-fill minimum of sum e^2 over every feasible W
              (convex separable, one multiplier, exact Fractions); it is
              attained, so it is the irreducible quant error of this pack.
  sse_max     chord-LP upper bound (convex terms <= their chords on each box,
              then a 2-variable LP); tight to within one weight's chord gap.
              Infinite for drop/clip without A (the dropped / clipped tail is
              unbounded), finite for absmean (S is pinned).
  energy      the same two tools on sum |W|^2, giving a certified relative
              bracket rel = SSE / ||W||^2 in [sse_min / e_max, sse_max / e_min]
              (JuniorHome docs/QUANT_ERROR.md: trit rel_l2 ~0.54 Gaussian).
Row trits [pack, consistent, pinned]: pack -1 on an I2_S code 3; pinned +1 if
sse_max - sse_min <= RATE * sse_max, 0 if sse_max is infinite, else -1.

It does not wrap, test, or re-derive any team file or earlier Winsor file
(JuniorQuant src/winsor_clip.py, src/i2sd_*; JuniorStock risk/winsor_quant_error.py,
quant/calibration/*.py). Weights given with --weights only feed the demo.

Pack: i2s 2-bit codes 0->0, 1->1, -1->2, MSB first, zero padded; code 3 is
invalid (pack trit -1). Close rule: key = sha256(prev || pack || text)[:16]
per row, GENESIS prev = "0" * 16, prev as ASCII hex, pack = the row's trits
packed, text = canonical JSON of the row. Rate lock: sha256(prev || pack ||
b"0.05").hexdigest() beside the final key. RATE = 0.05.

Run: python3 overlay_i2s_pack_only_error_bracket.py --selftest
     python3 overlay_i2s_pack_only_error_bracket.py --facts pack.json
       pack.json: {"scale": s, "n": n, "pack_hex": "..."} or
                  {"scale": s, "n0": n0, "n1": n1}; optional "eps", "amax", "prev"
     python3 overlay_i2s_pack_only_error_bracket.py --weights W.txt [--variant absmean]
Compatible engines: JuniorQuant, JuniorStock, BitNet-mlx, bitnetd, i2sd (same
I2_S code map and close rule).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from fractions import Fraction as F
from typing import Optional

RATE = 0.05
RATE_F = F(1, 20)
GENESIS = "0" * 16
VARIANTS = ("absmean", "drop", "clip")
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
TCP = {"bitnetd": "127.0.0.1:8765", "i2sd": "127.0.0.1:8767"}
CODE = {0: 0, 1: 1, -1: 2}
DECODE = {0: 0, 1: 1, 2: -1}


# ---------------------------------------------------------------- pack / close
def pack(trits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(trits), 4):
        b = 0
        chunk = list(trits[i:i + 4]) + [0] * (4 - len(trits[i:i + 4]))
        for t in chunk:
            b = (b << 2) | CODE[t]
        out.append(b)
    return bytes(out)


def unpack(data: bytes, n: int) -> tuple[Optional[list[int]], int]:
    """Return (trits or None if a code 3 appears in the first n, bad count)."""
    trits: list[int] = []
    bad = 0
    for b in data:
        for sh in (6, 4, 2, 0):
            if len(trits) + bad >= n:
                break
            c = (b >> sh) & 3
            if c == 3:
                bad += 1
            else:
                trits.append(DECODE[c])
    if len(trits) + bad < n:
        raise ValueError("pack shorter than n")
    return (None if bad else trits), bad


def close_rule(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def rate_lock(prev: str, pk: bytes) -> str:
    return hashlib.sha256(prev.encode() + pk + str(RATE).encode()).hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def fnum(x) -> Optional[float]:
    if x is None:
        return None
    return float(x)


# ---------------------------------------------------------------- core math
def sum_range(variant: str, n: int, gamma: F, amax: Optional[F]) -> tuple[F, Optional[F]]:
    """Exact [L, U] for S = sum|W| implied by the Winsor variant (U None = inf)."""
    lo = n * gamma
    if variant == "absmean":
        return lo, lo
    if amax is None:
        return lo, None
    if variant == "drop":
        tail = n // 16
    elif variant == "clip":
        tail = n - (-(-95 * n // 100))
    else:
        raise ValueError(variant)
    return lo, lo + tail * max(F(0), amax - gamma)


def feasible(n0: int, n1: int, theta: F, amax: Optional[F], L: F, U: Optional[F]):
    """Feasible S interval [a, b] (b None = inf) or None if empty."""
    s_lo = n1 * theta
    s_hi = None if (amax is None and n1 > 0) else n0 * theta + n1 * (amax if amax is not None else F(0))
    a = max(L, s_lo)
    b = s_hi if U is None else (U if s_hi is None else min(U, s_hi))
    if b is not None and a > b:
        return None
    return a, b


def eff_amax(n1: int, theta: F, amax: Optional[F], b: Optional[F]) -> Optional[F]:
    """Largest single nonzero |W|: declared A, tightened by S <= b."""
    if n1 == 0:
        return amax if amax is not None else theta
    cap = None if b is None else b - (n1 - 1) * theta
    if amax is None:
        return cap
    return amax if cap is None else min(amax, cap)


def _h(c: F, n0: int, n1: int, theta: F, A: F, tz: F, tn: F) -> F:
    return n0 * min(max(tz + c, F(0)), theta) + n1 * min(max(tn + c, theta), A)


def waterfill_min(n0, n1, theta, A, tz, tn, a, b):
    """min sum_Z (x - tz)^2 + sum_N (x - tn)^2, x_Z in [0,theta], x_N in [theta,A], S in [a,b]."""
    s0 = _h(F(0), n0, n1, theta, A, tz, tn)
    S = max(a, s0 if b is None else min(b, s0))
    bps = sorted({-tz, theta - tz, theta - tn, A - tn})
    hs = [_h(c, n0, n1, theta, A, tz, tn) for c in bps]
    c = bps[0] if S <= hs[0] else bps[-1]
    for i in range(len(bps) - 1):
        if hs[i] <= S <= hs[i + 1]:
            c = bps[i] if hs[i + 1] == hs[i] else bps[i] + (S - hs[i]) * (bps[i + 1] - bps[i]) / (hs[i + 1] - hs[i])
            break
    xz = min(max(tz + c, F(0)), theta)
    xn = min(max(tn + c, theta), A)
    val = n0 * (xz - tz) ** 2 + n1 * (xn - tn) ** 2
    return val, (xz, xn, S)


def chord_lp_max(n0, n1, theta, A, tz, tn, a, b):
    """Upper bound on sum of convex (x - t)^2 via chords, then a 2-var LP."""
    fz0, fz1 = tz ** 2, (theta - tz) ** 2
    fn0, fn1 = (theta - tn) ** 2, (A - tn) ** 2
    al = (fz1 - fz0) / theta if theta > 0 else F(0)
    be = (fn1 - fn0) / (A - theta) if A > theta else F(0)
    X, Y = n0 * theta, n1 * (A - theta)
    lo, hi = a - n1 * theta, b - n1 * theta
    cands = [(F(0), F(0)), (X, F(0)), (F(0), Y), (X, Y)]
    for t in (lo, hi):
        cands += [(F(0), t), (X, t - X), (t, F(0)), (t - Y, Y)]
    best = None
    for x, y in cands:
        if 0 <= x <= X and 0 <= y <= Y and lo <= x + y <= hi:
            v = n0 * fz0 + n1 * fn0 + al * x + be * y
            best = v if best is None or v > best else best
    gap = max(theta ** 2 / 4, ((A - theta) / 2) ** 2)
    return best, gap


def bracket(variant: str, n0: int, n1: int, s: F, eps: F, amax: Optional[F], gamma_scale: F = F(1)):
    n = n0 + n1
    theta = s / 2
    gamma = (s - eps) * gamma_scale
    L, U = sum_range(variant, n, gamma, amax)
    return n, theta, gamma, L, U, feasible(n0, n1, theta, amax, L, U)


def certify(variant: str, n0: int, n1: int, s: F, eps: F = F(0), amax: Optional[F] = None) -> dict:
    n, theta, gamma, L, U, fe = bracket(variant, n0, n1, s, eps, amax)
    if fe is None:
        ok_lo = bracket(variant, n0, n1, s, eps, amax, 1 - RATE_F)[5]
        ok_hi = bracket(variant, n0, n1, s, eps, amax, 1 + RATE_F)[5]
        trit = 0 if (ok_lo or ok_hi) else -1
        return {"variant": variant, "consistent": trit, "n": n, "n0": n0, "n1": n1,
                "scale": fnum(s), "gamma": fnum(gamma), "amax": fnum(amax),
                "amax_floor": None, "sse_min": None, "sse_max": None,
                "rel_floor": None, "rel_ceil": None, "trits": [1, trit, 0]}
    a, b = fe
    A = eff_amax(n1, theta, amax, b)
    floor_amax = None
    if n1 > 0:
        floor_amax = max(theta, (n * gamma - n0 * theta) / n1)
    if A is None:
        sse_max = e_max = None
        Aw = floor_amax if floor_amax is not None else theta
        gap = None
    else:
        Aw = A
        sse_max, gap = chord_lp_max(n0, n1, theta, A, F(0), s, a, b)
        e_max, _ = chord_lp_max(n0, n1, theta, A, F(0), F(0), a, b)
    # For the minima a finite box is needed; any cap >= the minimizer works.
    if A is None:
        big = max(Aw, s, a if b is None else b) * 4 + 1
        Aw = big
        b_eff = b
    else:
        b_eff = b
    sse_min, pt = waterfill_min(n0, n1, theta, Aw, F(0), s, a, b_eff)
    e_min, _ = waterfill_min(n0, n1, theta, Aw, F(0), F(0), a, b_eff)
    rel_floor = None if (e_max is None or e_max == 0) else sse_min / e_max
    rel_ceil = None if (sse_max is None or e_min == 0) else sse_max / e_min
    trit_tight = 0 if sse_max is None else (1 if sse_max - sse_min <= RATE_F * sse_max else -1)
    return {"variant": variant, "consistent": 1, "n": n, "n0": n0, "n1": n1,
            "scale": fnum(s), "gamma": fnum(gamma), "amax": fnum(amax),
            "amax_floor": fnum(floor_amax),
            "sse_min": fnum(sse_min), "sse_max": fnum(sse_max),
            "sse_min_per_w_s2": fnum(sse_min / (n * s * s)) if n and s else None,
            "sse_max_per_w_s2": fnum(sse_max / (n * s * s)) if (n and s and sse_max is not None) else None,
            "chord_gap": fnum(gap),
            "rel_floor": fnum(rel_floor), "rel_ceil": fnum(rel_ceil),
            "trits": [1, 1, trit_tight],
            "_exact": {"sse_min": sse_min, "sse_max": sse_max, "e_min": e_min,
                       "e_max": e_max, "amax_floor": floor_amax, "argmin": pt}}


def report(n0: int, n1: int, s: F, eps: F = F(0), amax: Optional[F] = None,
           prev: str = GENESIS, pack_trit: int = 1) -> dict:
    key = prev
    rows = []
    for v in VARIANTS:
        row = certify(v, n0, n1, s, eps, amax)
        row.pop("_exact", None)
        row["trits"][0] = pack_trit
        pk = pack(row["trits"])
        key = close_rule(key, pk, canon(row))
        row["key"] = key
        rows.append(row)
    pk_all = pack([r["consistent"] for r in rows])
    return {"tool": "overlay_i2s_pack_only_error_bracket", "rate": RATE,
            "roadmap": "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: I2_S 2B4T only if GGUF already on disk",
            "ports": list(PORTS), "tcp_named_not_dialed": TCP,
            "consistent_variants": [r["variant"] for r in rows if r["consistent"] == 1],
            "rows": rows, "key": key, "rate_lock": rate_lock(key, pk_all),
            "flags": {"mint": False, "socket": False, "layer_one": False,
                      "fetch": False, "ue5_launch": False}}


# ---------------------------------------------------------------- reference gammas
def gamma_of(variant: str, w: list[float]) -> F:
    a = sorted(abs(F(x)) for x in w)
    n = len(a)
    if variant == "absmean":
        return sum(a, F(0)) / n
    if variant == "drop":
        m = n - n // 16
        return sum(a[:m], F(0)) / m
    p = a[-(-95 * n // 100) - 1]
    return sum((min(x, p) for x in a), F(0)) / n


def quantize(w: list[float], s: F) -> list[int]:
    th = s / 2
    return [0 if abs(F(x)) <= th else (1 if x > 0 else -1) for x in w]


def true_stats(w: list[float], s: F, trits: list[int]):
    sse = sum(((abs(F(x)) - s * abs(t)) ** 2 for x, t in zip(w, trits)), F(0))
    en = sum((F(x) ** 2 for x in w), F(0))
    return sse, en, max(abs(F(x)) for x in w)


# ---------------------------------------------------------------- selftest
def selftest() -> str:
    assert pack([1, 0, -1, -1, 1, 0, 1, 0]) == bytes.fromhex("4a44")
    tr, bad = unpack(bytes.fromhex("4a44"), 8)
    assert tr == [1, 0, -1, -1, 1, 0, 1, 0] and bad == 0
    tr, bad = unpack(bytes([0b11000000]), 3)
    assert tr is None and bad == 1
    assert close_rule(GENESIS, b"\x02", "x") == hashlib.sha256(b"0" * 16 + b"\x02x").hexdigest()[:16]

    rng = random.Random(20261019)
    checked = 0
    for trial in range(12):
        n = rng.choice([64, 97, 128, 256])
        w = [rng.gauss(0.0, 1.0) for _ in range(n)]
        for _ in range(trial % 4):
            w[rng.randrange(n)] = rng.choice([-1, 1]) * rng.uniform(4.0, 12.0)
        w = [float(x) for x in w]
        for v in VARIANTS:
            g = gamma_of(v, w)
            for eps in (F(0), F(1, 100000)):
                s = g + eps
                t = quantize(w, s)
                n1 = sum(1 for x in t if x)
                n0 = n - n1
                sse, en, amax = true_stats(w, s, t)
                for A in (None, amax):
                    r = certify(v, n0, n1, s, eps, A)
                    assert r["consistent"] == 1, (v, A, r)
                    ex = r["_exact"]
                    assert ex["sse_min"] <= sse, (v, "sse_min")
                    if ex["sse_max"] is not None:
                        assert sse <= ex["sse_max"], (v, "sse_max")
                    if ex["e_max"] is not None:
                        assert en <= ex["e_max"]
                    assert ex["e_min"] <= en
                    assert ex["amax_floor"] <= amax
                    if r["rel_floor"] is not None:
                        assert r["rel_floor"] <= float(sse / en) * (1 + 1e-12)
                    if r["rel_ceil"] is not None:
                        assert float(sse / en) <= r["rel_ceil"] * (1 + 1e-12)
                    if v != "absmean" and A is None:
                        assert ex["sse_max"] is None
                    checked += 1
    # absmean without amax is still finite (S is pinned)
    r = certify("absmean", 40, 88, F(1))
    assert r["sse_max"] is not None and r["_exact"]["sse_max"] >= r["_exact"]["sse_min"]
    # sse_min is attained: build W at the water-fill point and recheck.
    n0, n1, s = 30, 98, F(3, 2)
    r = certify("absmean", n0, n1, s)
    xz, xn, S = r["_exact"]["argmin"]
    assert n0 * xz + n1 * xn == (n0 + n1) * s  # S = n gamma, eps 0
    assert n0 * xz ** 2 + n1 * (xn - s) ** 2 == r["_exact"]["sse_min"]
    assert xz <= s / 2 < xn
    # all-zero pack with a positive scale fits no Winsor gamma
    for v in VARIANTS:
        assert certify(v, 128, 0, F(1))["consistent"] == -1
    # RATE slack: s=1, n0=n1=50, A=7/5 gives S_hi=95 < n gamma=100, but
    # 0.95 * 100 = 95 fits, so the trit is 0; A=1 gives S_hi=75, trit -1.
    assert certify("absmean", 50, 50, F(1), F(0), F(7, 5))["consistent"] == 0
    assert certify("absmean", 50, 50, F(1), F(0), F(1))["consistent"] == -1
    assert certify("absmean", 50, 50, F(1), F(0), F(2))["consistent"] == 1
    amax = max(abs(F(x)) for x in w)
    # drop and clip gammas never exceed absmean, so a drop pack checked as absmean
    # may be inconsistent but never crashes
    for v in VARIANTS:
        certify(v, 200, 56, gamma_of("drop", w), F(0), amax)
    rep1 = report(40, 88, F(1), amax=F(5))
    rep2 = report(40, 88, F(1), amax=F(5))
    assert rep1["key"] == rep2["key"] and len(rep1["key"]) == 16
    assert len(rep1["rate_lock"]) == 64
    return close_rule(rep1["key"], pack([1, 1, 1]), "selftest:%d" % checked)


# ---------------------------------------------------------------- cli
def _frac(x) -> Optional[F]:
    return None if x is None else F(str(x)) if isinstance(x, str) else F(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--facts")
    ap.add_argument("--weights")
    ap.add_argument("--variant", choices=VARIANTS, default="absmean")
    ap.add_argument("--prev", default=GENESIS)
    args = ap.parse_args(argv)
    if len(args.prev) != 16 or any(c not in "0123456789abcdef" for c in args.prev):
        raise SystemExit("--prev must be 16 hex chars")
    if args.selftest:
        print("selftest ok key=" + selftest())
        return 0
    if args.facts:
        with open(args.facts, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        s = _frac(d["scale"])
        eps = _frac(d.get("eps", 0))
        amax = _frac(d.get("amax"))
        prev = d.get("prev", args.prev)
        if "pack_hex" in d:
            tr, bad = unpack(bytes.fromhex(d["pack_hex"]), int(d["n"]))
            if tr is None:
                print(canon({"pack": -1, "bad_codes": bad, "note": "code 3 in I2_S pack"}))
                return 2
            n1 = sum(1 for x in tr if x)
            n0 = len(tr) - n1
        else:
            n0, n1 = int(d["n0"]), int(d["n1"])
        print(canon(report(n0, n1, s, eps, amax, prev)))
        return 0
    if args.weights:
        with open(args.weights, "r", encoding="utf-8") as fh:
            w = [float(x) for x in fh.read().split()]
        g = gamma_of(args.variant, w)
        t = quantize(w, g)
        n1 = sum(1 for x in t if x)
        amax = max(abs(F(x)) for x in w)
        sse, en, _ = true_stats(w, g, t)
        rep = report(len(w) - n1, n1, g, F(0), amax, args.prev)
        rep["demo_true"] = {"variant": args.variant, "sse": float(sse), "rel": float(sse / en)}
        print(canon(rep))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
