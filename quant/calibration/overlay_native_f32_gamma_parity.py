"""JuniorOS overlay native-vs-stdlib gamma parity certificate (f32 C engines).

Stdlib only, offline. Writes nothing outside stdout. Loads no shared library.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay" (Debian/Alpine +
os-release.junior + hardened bitnetd.service + juniorctl on PATH; I2_S 2B4T only
if GGUF already on disk). Quant Chief, pair A, slot 2026-10-20 02:20 MT, remaining
gaps only. This is the quant-error / Winsor-bound side. It is not the os-release,
PATH, distro, MemoryMax, unit-parity, restart/ordering, GGUF-gate, pack-only,
per-128-block or i2sd-install item; it adds no service file and no OS config.

The gap: an overlay host that runs `make -C rails/linux so` gets native
quantizers (libjunior_absmean.so, libjunior_winsor.so); a host without them
falls back to the stdlib engines (rails/linux/absmean_ffi.py picks "c" or
"python" by whether the .so exists). The two backends do not compute the same
gamma or the same rounding:

  engine                         gamma accumulation            eps     rounding
  junior_bitnet.math.absmean     f64 sum(|w|)/n                none    round() half-even
  rails/linux absmean.c (SSE2/   4 f32 SIMD lanes, lanes       +1e-7f  roundf() half-away,
    NEON, i.e. every amd64 and   summed in float, f32 tail       (f32)   x/g in f32
    arm64 Debian/Alpine build)   in double, (float)(acc/n)
  junior_bitnet.winsor.pack      f64 sum(min(|w|,P95))/n       +1e-7   round() half-even
  rails/linux winsor.c           scalar f32 running sum,        +1e-7f  roundf() half-away
                                 acc/(float)n                    (f32)   x/g in f32

(The Junior drop gamma, mean |W| after dropping the top 1/16, has no native
engine, so it is stdlib-only on every overlay host and is reported "python_only".)

So the same weights can give a different trit stream, a different I2_S pack and
a different close key depending on whether the overlay host compiled the libs.

New capability (versus every earlier Winsor & Quant Error file): an exact f32
emulation of both C engines in stdlib (every f32 op is done in f64 and rounded
once with struct 'f', which is correctly rounded for + - / because 53 >= 2*24+2),
plus a certified C-vs-stdlib parity bound for each native gamma:

  * certified radius of each engine's gamma around the exact rational gamma
    (Higham g_k = k u / (1 - k u); u32 = 2**-24, u64 = 2**-53; lane depth
    m_max + 2 for the SIMD path, n - 1 for winsor.c; (float)n is inexact once
    n > 2**24, and g_k is undefined once k u32 >= 1, which makes winsor.c
    "uncertified" for a 2B4T FFN matrix, n = 2560 * 6912 = 17,694,720);
  * at-risk weights: |w| inside the band where either engine's 0.5 decision
    could go the other way (f32 division rounding and the half-even vs
    half-away tie rule included, so exact ties are always at risk);
  * a certified |dSSE| bound (JuniorHome docs/QUANT_ERROR.md SSE) summed per
    weight over every reachable (gamma, trit) state, against RATE = 0.05;
  * the f32 stall cap: with every term < 2**e, a running f32 sum can never pass
    2**(e+24) + 2**(e+1), so winsor.c gamma is provably biased low once the exact
    clipped sum exceeds that cap (reported for any n with --project).

Verdict trit per native gamma (absmean, clip): +1 no weight at risk (the C and
stdlib trit streams are provably identical), 0 at risk but certified relative
dSSE <= RATE, -1 certified bound > RATE or radius uncertified. Observed C-vs-
stdlib flips (by emulation) must never exceed the certified at-risk count.

Close rule: sha256(prev || pack || text)[:16], hex strings concatenated as UTF-8.
Per native gamma: pack = I2_S 2-bit codes of the emulated C trits (0->0, 1->1,
-1->2, four per byte, MSB first, hex), text = gamma.hex() of the emulated C
gamma. Report key: pack = I2_S codes of the verdict trits, text = canonical
JSON of the report core. GENESIS prev = "0" * 16. Rate 0.05.

Compatible engines: JuniorLLM rails/linux absmean.c / winsor.c (emulated,
never loaded), junior_bitnet.math.absmean, junior_bitnet.winsor.pack,
BitNet-mlx, JuniorQuant, JuniorStock, bitnetd 127.0.0.1:8765 and i2sd
127.0.0.1:8767 (named, never dialed).
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only (nothing is bound). mint false. socket false. layer_one
false. fetch false. ue5_launch false. No model pull, no GGUF download, no
network, no team-module import.

Usage:
  python3 overlay_native_f32_gamma_parity.py --selftest
  python3 overlay_native_f32_gamma_parity.py --weights FILE.json [--prev HEX16] [--no-simd]
  python3 overlay_native_f32_gamma_parity.py --synthetic 8192 --seed 7
  python3 overlay_native_f32_gamma_parity.py --project 17694720
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
from fractions import Fraction

RATE = 0.05
GENESIS = "0" * 16
U32 = 2.0 ** -24
U64 = 2.0 ** -53
F32_MAX = 3.4028234663852886e38
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False,
         "ue5_launch": False, "bind": "127.0.0.1"}
PORTS = ["Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"]
TWO_B4T_FFN = 2560 * 6912

_PACK_F = struct.Struct("<f")


def f32(x: float) -> float:
    """Round an f64 to the nearest f32 (ties to even), as a C float store."""
    if x != x or abs(x) > F32_MAX:
        raise ValueError("value outside f32 range")
    return _PACK_F.unpack(_PACK_F.pack(x))[0]


EPS32 = f32(1e-7)


def roundf(x: float) -> float:
    """C roundf: half away from zero."""
    return math.copysign(math.floor(abs(x) + 0.5), x) if abs(x) < 2.0 ** 23 else x


def clip_trit(q: float) -> int:
    return -1 if q < -1 else (1 if q > 1 else int(q))


# ---------------------------------------------------------------- engines

def c_absmean(xs: list[float], simd: bool = True) -> tuple[float, list[int]]:
    """Exact emulation of rails/linux/absmean.c junior_absmean."""
    n = len(xs)
    i = 0
    acc = 0.0
    if simd:
        lanes = [0.0, 0.0, 0.0, 0.0]
        while i + 4 <= n:
            for j in range(4):
                lanes[j] = f32(lanes[j] + abs(xs[i + j]))
            i += 4
        acc = f32(f32(f32(lanes[0] + lanes[1]) + lanes[2]) + lanes[3])
    while i < n:
        acc = acc + abs(xs[i])
        i += 1
    g = f32(f32(acc / float(n)) + EPS32)
    return g, [clip_trit(roundf(f32(x / g))) for x in xs]


def c_winsor_index(n: int) -> int:
    idx = int((95.0 / 100.0) * (n - 1))
    return min(idx, n - 1)


def c_winsor(xs: list[float]) -> tuple[float, list[int], float]:
    """Exact emulation of rails/linux/winsor.c junior_winsor."""
    n = len(xs)
    absv = sorted(abs(x) for x in xs)
    tau = absv[c_winsor_index(n)]
    acc = 0.0
    for a in absv:
        acc = f32(acc + (a if a < tau else tau))
    g = f32(f32(acc / f32(float(n))) + EPS32)
    return g, [clip_trit(roundf(f32(x / g))) for x in xs], tau


def py_absmean(xs: list[float]) -> tuple[float, list[int]]:
    """Same arithmetic as junior_bitnet.math.absmean (f64, no eps, half-even)."""
    d = (sum(abs(x) for x in xs) / len(xs)) or 1.0
    return d, [max(-1, min(1, int(round(x / d)))) for x in xs]


def py_winsor(xs: list[float]) -> tuple[float, list[int]]:
    """Same arithmetic as junior_bitnet.winsor.pack (f64, +1e-7, half-even)."""
    a = [abs(float(w)) for w in xs]
    s = sorted(a)
    tau = s[min(len(s) - 1, max(0, math.floor(0.95 * (len(s) - 1))))]
    gamma = sum(min(v, tau) for v in a) / max(1, len(a))
    g = gamma + 1e-7
    return g, [max(-1, min(1, int(round(float(w) / g)))) for w in xs]


def drop_gamma(xs: list[float]) -> float:
    a = sorted(abs(x) for x in xs)
    keep = a[: len(a) - len(a) // 16] or a
    return math.fsum(keep) / len(keep)


# ---------------------------------------------------------------- bounds

def gamma_k(k: int, u: float) -> float:
    """Higham g_k; inf when k*u >= 1 (no certificate)."""
    ku = k * u
    return math.inf if ku >= 1.0 else ku / (1.0 - ku)


def radius_c_absmean(n: int, simd: bool) -> float:
    """Relative radius: |g_c - (g + eps)| <= R * g + eps * u32."""
    if simd:
        m_max = n // 4  # terms per SIMD lane
        body = gamma_k(m_max + 2, U32) if n >= 4 else 0.0
        tail = n % 4
        rel = (1.0 + body) * (1.0 + U64) ** (tail + 1) * (1.0 + U32) ** 2 - 1.0
    else:
        rel = (1.0 + gamma_k(n, U64)) * (1.0 + U64) * (1.0 + U32) ** 2 - 1.0
    return rel


def radius_c_winsor(n: int) -> float:
    body = gamma_k(max(0, n - 1), U32)
    nf = 1.0 if n <= 2 ** 24 else 1.0 / (1.0 - U32)
    return (1.0 + body) * nf * (1.0 + U32) ** 3 - 1.0


def radius_py(n: int) -> float:
    """Covers recursive and compensated (CPython >= 3.12) float sum, plus /n."""
    return (1.0 + gamma_k(n, U64) + 2.0 * U64) * (1.0 + U64) - 1.0


def stall_cap(tau: float) -> float:
    """Every term < 2**e  =>  running f32 sum never exceeds 2**(e+24) + 2**(e+1)."""
    if tau <= 0:
        return 0.0
    e = math.floor(math.log2(tau)) + 1
    return 2.0 ** (e + 24) + 2.0 ** (e + 1)


def interval(g_exact: Fraction, eps: float, rel: float, eps_slack: float) -> tuple[float, float]:
    if math.isinf(rel):
        return (0.0, math.inf)
    c = float(g_exact) + eps
    r = float(g_exact) * rel + eps_slack
    pad = abs(c) * 4 * U64  # float(Fraction) and the sum above are rounded once each
    return (max(0.0, c - r - pad), c + r + pad)


def certify(xs: list[float], g_lo: float, g_hi: float) -> tuple[int, float, list[int]]:
    """At-risk count, certified |dSSE| bound, at-risk indices.

    A weight is safe only if every gamma in [g_lo, g_hi] gives the same trit in
    both engines: |w| < 0.5 g_lo (1 - 2**-24) -> 0, |w| > 0.5 g_hi (1 + 2**-24)
    -> sign(w). The band covers f32/f64 division rounding and either tie rule.
    """
    lo_cut = 0.5 * g_lo * (1.0 - 2.0 ** -24)
    hi_cut = 0.5 * g_hi * (1.0 + 2.0 ** -24)
    k = 0
    risky: list[int] = []
    dsse = 0.0
    for i, w in enumerate(xs):
        a = abs(w)
        if a < lo_cut:
            continue  # trit 0 in both, error w^2 in both
        def sq_range() -> tuple[float, float]:
            hi = max((a - g_lo) ** 2, (a - g_hi) ** 2)
            lo = 0.0 if g_lo <= a <= g_hi else min((a - g_lo) ** 2, (a - g_hi) ** 2)
            return lo, hi
        lo1, hi1 = sq_range()
        if a > hi_cut:
            dsse += hi1 - lo1
            continue
        k += 1
        risky.append(i)
        dsse += max(hi1, a * a) - min(lo1, a * a)
    return k, dsse * (1.0 + 1e-12), risky


def sse(xs: list[float], g: float, t: list[int]) -> float:
    return math.fsum((w - g * q) ** 2 for w, q in zip(xs, t))


# ---------------------------------------------------------------- close rule

def i2s_pack(trits: list[int]) -> str:
    code = {0: 0, 1: 1, -1: 2}
    out = bytearray()
    for i in range(0, len(trits), 4):
        b = 0
        chunk = trits[i:i + 4]
        for j in range(4):
            b = (b << 2) | (code[chunk[j]] if j < len(chunk) else 0)
        out.append(b)
    return out.hex()


def close(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- report

def verdict(k: int, rel_bound: float, certified: bool) -> int:
    if not certified or rel_bound > RATE:
        return -1
    return 1 if k == 0 else 0


def analyze(xs_in: list[float], prev: str = GENESIS, simd: bool = True) -> dict:
    if len(xs_in) < 2:
        raise ValueError("need at least 2 weights")
    non_f32 = sum(1 for x in xs_in if f32(x) != x)
    xs = [f32(x) for x in xs_in]  # what absmean_ffi hands to C (c_float array)
    n = len(xs)
    out: dict = {"n": n, "simd": simd, "inputs_not_f32_exact": non_f32, "rate": RATE,
                 "flags": FLAGS, "ports": PORTS}

    # absmean
    g_ex = sum((Fraction(abs(x)) for x in xs), Fraction(0)) / n
    rc = radius_c_absmean(n, simd)
    rp = radius_py(n)
    gc, tc = c_absmean(xs, simd)
    gp, tp = py_absmean(xs)
    lo_c, hi_c = interval(g_ex, EPS32, rc, EPS32 * U32)
    lo_p, hi_p = interval(g_ex, 0.0, rp, 0.0)
    assert lo_c <= gc <= hi_c, "C absmean outside certified radius"
    assert lo_p <= gp <= hi_p, "stdlib absmean outside certified radius"
    g_lo, g_hi = min(lo_c, lo_p), max(hi_c, hi_p)
    cert = not math.isinf(g_hi)
    k, dsse, _ = certify(xs, g_lo, g_hi) if cert else (n, math.inf, [])
    sp = sse(xs, gp, tp)
    flips = sum(1 for a, b in zip(tc, tp) if a != b)
    assert flips <= k, "observed flips exceed certified at-risk count"
    rel_b = dsse / sp if sp > 0 else (0.0 if dsse == 0 else math.inf)
    obs = abs(sse(xs, gc, tc) - sp) / sp if sp > 0 else 0.0
    assert obs <= rel_b * (1 + 1e-9) + 1e-15, "observed dSSE exceeds bound"
    v_abs = verdict(k, rel_b, cert)
    out["absmean"] = {
        "gamma_exact": float(g_ex), "gamma_c": gc, "gamma_py": gp,
        "rel_radius_c": rc, "rel_radius_py": rp, "certified": cert,
        "at_risk": k, "flips_observed": flips, "rel_dsse_bound": rel_b,
        "rel_dsse_observed": obs, "verdict": v_abs,
        "c_err_rel": abs(gc - EPS32 - float(g_ex)) / float(g_ex) if g_ex else 0.0,
    }

    # P95 clip
    absv = sorted(abs(x) for x in xs)
    idx = c_winsor_index(n)
    assert idx == min(n - 1, max(0, math.floor(0.95 * (n - 1)))), "P95 index drift"
    tau = absv[idx]
    gw_ex = sum((Fraction(min(a, tau)) for a in absv), Fraction(0)) / n
    rcw = radius_c_winsor(n)
    gcw, tcw, tau_c = c_winsor(xs)
    gpw, tpw = py_winsor(xs)
    eps64 = 1e-7
    lo_cw, hi_cw = interval(gw_ex, EPS32, rcw, EPS32 * U32)
    lo_pw, hi_pw = interval(gw_ex, eps64, rp, eps64 * U64)
    certw = not math.isinf(rcw)
    if certw:
        assert lo_cw <= gcw <= hi_cw, "C winsor outside certified radius"
    assert lo_pw <= gpw <= hi_pw, "stdlib winsor outside certified radius"
    kw, dssew, _ = certify(xs, min(lo_cw, lo_pw), max(hi_cw, hi_pw)) if certw else (n, math.inf, [])
    spw = sse(xs, gpw, tpw)
    flipsw = sum(1 for a, b in zip(tcw, tpw) if a != b)
    if certw:
        assert flipsw <= kw, "observed winsor flips exceed certified at-risk count"
    rel_bw = dssew / spw if spw > 0 else (0.0 if dssew == 0 else math.inf)
    obsw = abs(sse(xs, gcw, tcw) - spw) / spw if spw > 0 else 0.0
    if certw:
        assert obsw <= rel_bw * (1 + 1e-9) + 1e-15, "observed winsor dSSE exceeds bound"
    cap = stall_cap(tau)
    v_clip = verdict(kw, rel_bw, certw)
    out["clip_p95"] = {
        "tau": tau, "gamma_exact": float(gw_ex), "gamma_c": gcw, "gamma_py": gpw,
        "rel_radius_c": rcw, "certified": certw, "at_risk": kw,
        "flips_observed": flipsw, "rel_dsse_bound": rel_bw,
        "rel_dsse_observed": obsw, "verdict": v_clip,
        "f32_stall_cap": cap,
        "stall_bias_certain": float(gw_ex) * n > cap,
    }
    out["drop"] = {"gamma": drop_gamma(xs), "native": "python_only"}

    k1 = close(prev, i2s_pack(tc), gc.hex())
    k2 = close(k1, i2s_pack(tcw), gcw.hex())
    core = {"n": n, "simd": simd, "verdicts": [v_abs, v_clip],
            "at_risk": [k, kw], "gamma_c": [gc.hex(), gcw.hex()], "rate": RATE}
    text = json.dumps(core, sort_keys=True, separators=(",", ":"))
    out["keys"] = {"absmean": k1, "clip_p95": k2,
                   "report": close(k2, i2s_pack([v_abs, v_clip]), text)}
    out["verdicts"] = {"absmean": v_abs, "clip_p95": v_clip, "drop": "python_only"}
    return out


def project(n: int, simd: bool = True, tau_over_mean: float = 2.5) -> dict:
    """Tensor-free projection for an n-weight matrix (no emulation)."""
    rc = radius_c_absmean(n, simd)
    rw = radius_c_winsor(n)
    # stall: sum(min(|w|,tau)) = mean_clip * n; cap <= 2**25 * tau (tau < 2**e <= 2 tau)
    n_stall = math.ceil((2.0 ** 25 + 2.0 ** 2) * tau_over_mean)
    return {
        "n": n, "simd": simd,
        "absmean_rel_radius_c": rc,
        "absmean_certified_within_rate": rc <= RATE,
        "winsor_rel_radius_c": rw,
        "winsor_certified": not math.isinf(rw),
        "float_n_exact": n <= 2 ** 24,
        "winsor_stall_bias_certain_from_n": n_stall,
        "tau_over_mean_assumed": tau_over_mean,
        "largest_n_absmean_simd_within_rate": _largest_n(lambda m: radius_c_absmean(m, simd)),
        "largest_n_winsor_within_rate": _largest_n(radius_c_winsor),
    }


def _largest_n(fn) -> int:
    lo, hi = 2, 1 << 40
    if fn(lo) > RATE:
        return 0
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fn(mid) <= RATE:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------- selftest

def synth(n: int, seed: int, heavy: bool = False) -> list[float]:
    """Irwin-Hall bell (+ optional rational heavy tail); libm-free, f32-exact."""
    r = random.Random(seed)
    xs = []
    for _ in range(n):
        z = (r.random() + r.random() + r.random() + r.random() - 2.0) * 0.03
        if heavy:
            z *= 1.0 + 6.0 * r.random() ** 6
        xs.append(f32(z))
    return xs


def selftest() -> int:
    # f32 helpers
    assert f32(0.1) != 0.1 and f32(f32(0.1)) == f32(0.1)
    assert roundf(0.5) == 1.0 and roundf(-0.5) == -1.0 and round(0.5) == 0
    assert roundf(2.5) == 3.0 and round(2.5) == 2
    # stall: at 2**24, adding 1.0 (exactly half an ulp) ties to even and stays
    acc = 2.0 ** 24
    for _ in range(8):
        acc = f32(acc + 1.0)
    assert acc == 2.0 ** 24
    tau = 0.75  # < 2**0
    acc = stall_cap(tau)
    acc0 = f32(acc)
    for _ in range(8):
        acc = f32(acc + tau)
    assert acc == acc0, "stall cap violated"
    # I2_S pack + close rule
    assert i2s_pack([1, -1, 0, 1]) == "61" and i2s_pack([-1]) == "80"
    assert close(GENESIS, "61", "x") == hashlib.sha256(
        (GENESIS + "61x").encode()).hexdigest()[:16]
    # exact tie: python half-even gives 0, C roundf gives 1 on the same ratio
    g = 0.25
    w = 0.125
    assert round(w / g) == 0 and roundf(f32(w / g)) == 1.0
    k_tie, _, risky = certify([w, 0.01, 0.4], g, g)
    assert k_tie == 1 and risky == [0], "exact tie must be at risk"
    # radius checks
    assert radius_c_winsor(TWO_B4T_FFN) == math.inf
    assert radius_c_absmean(4096, True) < radius_c_winsor(4096)
    # end-to-end, bell and heavy tail, SIMD and scalar absmean
    reps = []
    for n, seed, heavy, simd in ((4096, 7, False, True), (6003, 11, True, True),
                                 (4099, 5, False, False)):
        rep = analyze(synth(n, seed, heavy), GENESIS, simd)
        assert rep["absmean"]["certified"] and rep["clip_p95"]["certified"]
        for kname in ("absmean", "clip_p95"):
            assert rep[kname]["flips_observed"] <= rep[kname]["at_risk"]
            assert rep[kname]["verdict"] in (-1, 0, 1)
        reps.append(rep)
    again = analyze(synth(4096, 7), GENESIS, True)
    assert again["keys"] == reps[0]["keys"], "key not deterministic"
    pr = project(TWO_B4T_FFN)
    assert not pr["winsor_certified"] and not pr["float_n_exact"]
    assert pr["largest_n_winsor_within_rate"] < pr["largest_n_absmean_simd_within_rate"]
    summary = {
        "selftest": "pass",
        "key": reps[0]["keys"]["report"],
        "verdicts": [r["verdicts"] for r in reps],
        "at_risk": [[r["absmean"]["at_risk"], r["clip_p95"]["at_risk"]] for r in reps],
        "flips": [[r["absmean"]["flips_observed"], r["clip_p95"]["flips_observed"]] for r in reps],
        "project_2b4t_ffn": pr,
        "flags": FLAGS, "ports": PORTS, "rate": RATE,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--weights", help="JSON list of floats (local file only)")
    ap.add_argument("--synthetic", type=int)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("--project", type=int)
    ap.add_argument("--prev", default=GENESIS)
    ap.add_argument("--no-simd", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.project:
        print(json.dumps(project(a.project, not a.no_simd), indent=2, sort_keys=True))
        return 0
    if a.weights:
        with open(a.weights, "r", encoding="utf-8") as fh:
            xs = [float(v) for v in json.load(fh)]
    elif a.synthetic:
        xs = synth(a.synthetic, a.seed, a.heavy)
    else:
        ap.print_help()
        return 2
    rep = analyze(xs, a.prev, not a.no_simd)
    print(json.dumps(rep, indent=2, sort_keys=True))
    worst = min(rep["verdicts"]["absmean"], rep["verdicts"]["clip_p95"])
    return {1: 0, 0: 0, -1: 1}[worst]


if __name__ == "__main__":
    sys.exit(main())
