"""JuniorOS overlay bit-plane matvec (rails/linux bitplane.c) f32 error certificate.

Stdlib only, offline. Writes nothing outside stdout. Loads no shared library.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay" (Debian/Alpine +
os-release.junior + hardened bitnetd.service + juniorctl on PATH; I2_S 2B4T only
if GGUF already on disk). Quant Chief, pair A, remaining gaps only. This is the
quant-error / Winsor-bound side. It is not os-release, PATH, distro, MemoryMax,
unit parity, restart/ordering, sandbox reach, seccomp, the GGUF gate, the
pack-only bracket, the per-128-block scale, the i2sd install closure, bitnetd
second-tier hardening, or native gamma parity: those certify how the overlay
gets gamma and the trits. Nothing so far certifies what the overlay does with
them at inference time.

The gap: `make -C rails/linux so` (overlay Makefile) also builds
libjunior_bitplane.so, and rails/linux/bitplane_ffi.py feeds it from bitnetd's
side of the overlay. The kernel is

    pos = sum_{t=+1} x_j   (scalar f32 running sum)
    neg = sum_{t=-1} x_j   (scalar f32 running sum)
    y   = gamma * (pos - neg)          (f32 subtract, f32 multiply)

with x and gamma cast to f32 at the FFI boundary. Splitting the row into two
same-sign planes and subtracting at the end means the rounding error scales with
A = sum_{t!=0} |x_j| while the answer scales with |D| = |sum t_j x_j|, so the
relative error is amplified by kappa = A / |D| (cancellation). No earlier file
bounds this kernel.

New capability (versus every earlier Winsor & Quant Error file): an exact f32
emulation of junior_bitplane (every f32 op done in f64 and rounded once through
struct 'f', correctly rounded since 53 >= 2*24+2) plus a rigorous per-row bound

    e1 = g_{p-1} A_pos + g_{n-1} A_neg              (Higham g_k = k u/(1-k u))
    e2 = e1 + u (|D| + e1)                          (subtract)
    E  = g32 e2 + u g32 (|D| + e2) + |g32 - g64| |D| + g64 sum_{t!=0}|x - x32|
                                                    (multiply, both FFI casts)

u = 2**-24, g_k undefined once k u >= 1 (row reported uncertified). The kernel
error is then compared with the weight-quant error of the same row,
|W.x - gamma T.x| (JuniorHome docs/QUANT_ERROR.md), for each docs/WINSOR.md
gamma: BitNet absmean mean|W|, Junior drop (mean |W| after dropping the top
1/16), and P95 clip mean min(|W|, P95). Trits are clip(round(W/gamma), -1, 1).

Verdict trit per gamma: +1 every row's certified E <= RATE * quant error (the
f32 kernel is provably negligible next to ternarization), 0 the bound misses
RATE on some row but the emulated error stays within it, -1 the emulated error
itself exceeds RATE * quant error on some row, or a row is uncertified.
Emulated error must never exceed E (self-test asserts it).

Close rule: sha256(prev || pack || text)[:16], hex strings concatenated as UTF-8.
Per gamma: pack = I2_S 2-bit codes of the row-major trits (0->0, 1->1, -1->2,
four per byte, MSB first, zero-padded, as rails/linux i2s_pack.c), text = the
emulated f32 outputs as float.hex() joined by ",". Keys chain absmean -> drop ->
clip. Report key: pack = I2_S codes of the three verdict trits, text = canonical
JSON of the report core. GENESIS prev = "0" * 16. Rate 0.05.

Compatible engines: JuniorLLM rails/linux bitplane.c via bitplane_ffi.py
(emulated, never loaded), junior_osai.kernels.bitplane (numpy f32 planes, same
two-plane split), junior_bitnet absmean/winsor packers, BitNet-mlx, JuniorQuant,
JuniorStock, bitnetd 127.0.0.1:8765 and i2sd 127.0.0.1:8767 (named, never dialed).
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only (nothing is bound). mint false. socket false. layer_one
false. fetch false. ue5_launch false. No model pull, no GGUF download, no
network, no team-module import.

Usage:
  python3 overlay_bitplane_f32_matvec_bound.py --selftest
  python3 overlay_bitplane_f32_matvec_bound.py --synthetic 16x2560 --seed 7 [--heavy]
  python3 overlay_bitplane_f32_matvec_bound.py --json FILE.json [--prev HEX16]
      (FILE.json: {"W": [[...], ...], "x": [...]})
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
F32_MAX = 3.4028234663852886e38
EPS = 1e-7
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False,
         "ue5_launch": False, "bind": "127.0.0.1"}
PORTS = ["Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"]
ENDPOINTS = {"bitnetd": "127.0.0.1:8765", "i2sd": "127.0.0.1:8767"}
_F = struct.Struct("<f")


def f32(v: float) -> float:
    if v != v or abs(v) > F32_MAX:
        raise ValueError("value outside f32 range")
    return _F.unpack(_F.pack(v))[0]


def g_k(k: int, u: float = U32) -> float:
    if k <= 0:
        return 0.0
    if k * u >= 1.0:
        return math.inf
    return (k * u) / (1.0 - k * u)


# ---- gammas (docs/WINSOR.md) ------------------------------------------------

def gamma_absmean(ws: list[float]) -> float:
    return math.fsum(abs(w) for w in ws) / len(ws) + EPS


def gamma_drop(ws: list[float]) -> float:
    a = sorted(abs(w) for w in ws)
    keep = a[: len(a) - len(a) // 16] or a
    return math.fsum(keep) / len(keep) + EPS


def gamma_clip(ws: list[float]) -> float:
    a = sorted(abs(w) for w in ws)
    p95 = a[min(len(a) - 1, int(math.ceil(0.95 * len(a))) - 1)]
    return math.fsum(min(v, p95) for v in a) / len(a) + EPS


GAMMAS = (("absmean", gamma_absmean), ("drop", gamma_drop), ("clip", gamma_clip))


def ternarize(ws: list[float], g: float) -> list[int]:
    return [max(-1, min(1, round(w / g))) for w in ws]


# ---- kernel emulation and bound --------------------------------------------

def bitplane_row(t: list[int], x32: list[float], g32: float) -> float:
    """Exact emulation of one junior_bitplane row (scalar f32, in order)."""
    pos = 0.0
    neg = 0.0
    for tj, xj in zip(t, x32):
        if tj > 0:
            pos = f32(pos + xj)
        elif tj < 0:
            neg = f32(neg + xj)
    return f32(g32 * f32(pos - neg))


def row_bound(t: list[int], x: list[float], x32: list[float],
              g64: float, g32: float) -> dict:
    p = sum(1 for v in t if v > 0)
    n = sum(1 for v in t if v < 0)
    a_pos = math.fsum(abs(xj) for tj, xj in zip(t, x32) if tj > 0)
    a_neg = math.fsum(abs(xj) for tj, xj in zip(t, x32) if tj < 0)
    d_exact = sum((Fraction(xj) * tj for tj, xj in zip(t, x32) if tj), Fraction(0))
    d = abs(float(d_exact))
    gp, gn = g_k(p - 1), g_k(n - 1)
    certified = math.isfinite(gp) and math.isfinite(gn)
    # inflate fsum by 1 + 2**-50 so the f64 bookkeeping is itself covered
    s = 1.0 + 2.0 ** -50
    e1 = (gp * a_pos + gn * a_neg) * s if certified else math.inf
    e2 = e1 + U32 * (d + e1)
    cast_x = math.fsum(abs(a - b) for tj, a, b in zip(t, x, x32) if tj)
    e = (g32 * e2 + U32 * g32 * (d + e2) + abs(g32 - g64) * d + g64 * cast_x) * s
    y_ref = g64 * math.fsum(tj * xj for tj, xj in zip(t, x) if tj)
    amp = (a_pos + a_neg) / d if d > 0 else math.inf
    return {"p": p, "n": n, "bound": e, "certified": certified,
            "y_ref": y_ref, "kappa": amp}


def i2s_pack(trits: list[int]) -> str:
    out = bytearray()
    acc = bits = 0
    for v in trits:
        acc = (acc << 2) | (0 if v == 0 else (1 if v == 1 else 2))
        bits += 2
        if bits == 8:
            out.append(acc)
            acc = bits = 0
    if bits:
        out.append((acc << (8 - bits)) & 0xFF)
    return out.hex()


def close(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


def analyze(W: list[list[float]], x: list[float], prev: str = GENESIS) -> dict:
    if not W or any(len(r) != len(x) for r in W) or not x:
        raise ValueError("W must be m x k with k == len(x) > 0")
    x32 = [f32(v) for v in x]
    flat = [w for r in W for w in r]
    out: dict = {"m": len(W), "k": len(x), "rate": RATE, "gammas": {}}
    key = prev
    verdicts = []
    for name, fn in GAMMAS:
        g64 = fn(flat)
        g32 = f32(g64)
        rows = []
        trits_all: list[int] = []
        ys: list[str] = []
        worst_bound_ratio = 0.0
        worst_obs_ratio = 0.0
        kappa_max = 0.0
        uncert = 0
        for r in W:
            t = ternarize(r, g64)
            trits_all.extend(t)
            y_hat = bitplane_row(t, x32, g32)
            ys.append(y_hat.hex())
            b = row_bound(t, x, x32, g64, g32)
            obs = abs(y_hat - b["y_ref"])
            if obs > b["bound"]:
                raise AssertionError("emulated error exceeds certified bound")
            q_err = abs(math.fsum(w * xj for w, xj in zip(r, x)) - b["y_ref"])
            denom = RATE * q_err
            br = b["bound"] / denom if denom > 0 else (0.0 if b["bound"] == 0 else math.inf)
            orat = obs / denom if denom > 0 else (0.0 if obs == 0 else math.inf)
            worst_bound_ratio = max(worst_bound_ratio, br)
            worst_obs_ratio = max(worst_obs_ratio, orat)
            if math.isfinite(b["kappa"]):
                kappa_max = max(kappa_max, b["kappa"])
            uncert += 0 if b["certified"] else 1
            rows.append(obs)
        if uncert or worst_obs_ratio > 1.0:
            v = -1
        elif worst_bound_ratio <= 1.0:
            v = 1
        else:
            v = 0
        verdicts.append(v)
        key = close(key, i2s_pack(trits_all), ",".join(ys))
        out["gammas"][name] = {
            "gamma": g64, "gamma_f32": g32,
            "worst_bound_over_rate_quant": round(worst_bound_ratio, 9),
            "worst_observed_over_rate_quant": round(worst_obs_ratio, 9),
            "max_observed_abs_err": max(rows),
            "kappa_max": round(kappa_max, 6),
            "uncertified_rows": uncert, "verdict": v, "key": key}
    core = {"m": out["m"], "k": out["k"], "rate": RATE, "verdicts": verdicts,
            "keys": [out["gammas"][n]["key"] for n, _ in GAMMAS]}
    out["verdicts"] = verdicts
    out["report_key"] = close(key, i2s_pack(verdicts),
                              json.dumps(core, sort_keys=True, separators=(",", ":")))
    out["flags"] = FLAGS
    out["ports"] = PORTS
    out["endpoints_named_not_dialed"] = ENDPOINTS
    return out


def synth(m: int, k: int, seed: int, heavy: bool = False) -> tuple[list[list[float]], list[float]]:
    rng = random.Random(seed)
    W = [[rng.gauss(0.0, 0.02) * (rng.paretovariate(2.5) if heavy else 1.0)
          for _ in range(k)] for _ in range(m)]
    x = [rng.gauss(0.0, 1.0) * (rng.paretovariate(1.8) if heavy else 1.0) for _ in range(k)]
    return W, x


def selftest() -> int:
    assert f32(0.1) != 0.1 and f32(f32(0.1)) == f32(0.1)
    assert g_k(0) == 0.0 and g_k(2 ** 24) == math.inf
    assert i2s_pack([1, -1, 0, 1]) == format(0b01100001, "02x")
    assert i2s_pack([-1]) == "80"
    assert close(GENESIS, "61", "x") == hashlib.sha256(
        (GENESIS + "61x").encode()).hexdigest()[:16]
    # hand row: exact small integers, no rounding anywhere -> zero error
    t = [1, -1, 0, 1]
    xs = [1.0, 2.0, 5.0, 4.0]
    assert bitplane_row(t, xs, 0.5) == 1.5
    b = row_bound(t, xs, xs, 0.5, 0.5)
    assert b["y_ref"] == 1.5 and b["bound"] >= 0.0
    # cancellation: big equal planes, tiny difference -> kappa explodes
    tc = [1, -1] * 512
    xc = [1.0 + 2.0 ** -20 if j % 2 == 0 else 1.0 for j in range(1024)]
    bc = row_bound(tc, xc, [f32(v) for v in xc], 1.0, 1.0)
    assert bc["kappa"] > 1e5
    # rigour on many shapes: analyze() raises if any emulated error > bound
    keys = []
    for (m, k, seed, heavy) in [(8, 256, 1, False), (8, 512, 2, True),
                                (4, 2560, 3, False), (4, 2560, 4, True)]:
        W, x = synth(m, k, seed, heavy)
        rep = analyze(W, x)
        assert all(v in (-1, 0, 1) for v in rep["verdicts"])
        for gname, _ in GAMMAS:
            gi = rep["gammas"][gname]
            assert gi["uncertified_rows"] == 0
            assert gi["max_observed_abs_err"] >= 0.0
        keys.append(rep["report_key"])
    W, x = synth(8, 256, 1)
    assert analyze(W, x)["report_key"] == keys[0]
    assert analyze(W, x, prev="f" * 16)["report_key"] != keys[0]
    assert len(set(keys)) == len(keys)
    assert FLAGS["bind"] == "127.0.0.1" and not any(
        FLAGS[f] for f in ("mint", "socket", "layer_one", "fetch", "ue5_launch"))
    print(json.dumps({"selftest": "pass", "keys": keys, "rate": RATE,
                      "ports": PORTS}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--synthetic", default=None, help="MxK, e.g. 16x2560")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.json:
        with open(a.json, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        W, x = d["W"], d["x"]
    elif a.synthetic:
        m, k = (int(v) for v in a.synthetic.lower().split("x"))
        W, x = synth(m, k, a.seed, a.heavy)
    else:
        ap.print_help()
        return 2
    print(json.dumps(analyze(W, x, a.prev), sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
