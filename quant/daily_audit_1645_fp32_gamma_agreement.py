"""16:45 daily audit: fp32 vs fp64 Winsor gamma agreement for BitNet-mlx and JuniorQuant.

Roadmap item: JuniorHome docs/ROADMAP.md, Automations (current) ON:
"16:45 daily audit", named for pair A by Quant Chief on 2026-10-23 14:10 MT
("remaining engines BitNet-mlx / JuniorQuant"; landed stocksnode 4eb7e33,
web3node 5eea5d6, Handshake c1b5826). The audit already records each port's
quant error against its analytic bound (JuniorStock b15f1f0). What it does
not record is whether two engines that compute the same gamma agree.
BitNet-mlx runs in float32 (MLX), JuniorQuant in float64, and the i2sd
userspace (127.0.0.1:8767) trusts whichever gamma it is handed. This file
adds that check: per engine, port and gamma variant, the fp32 gamma must sit
inside a derived rounding bound of the fp64 reference, and every ternary code
that flips between the two precisions must lie inside the rounding band of a
decision threshold (|w| / g = 0.5). An unexplained flip is a real defect,
not rounding.

New capability. It never imports, runs, wraps, tests, rewrites or re-derives
any team file or any earlier Winsor & Quant Error file (JuniorQuant
src/winsor_clip.py, src/i2sd_quant_budget.py, src/i2sd_residual_bound.py,
src/i2sd_act_accum_bound.py, src/weekly_audit_*; JuniorStock
risk/winsor_quant_error.py, security/winsor_outlier_health.py,
quant/calibration/winsor_night_drift.py, quant/daily_audit_1645_quant_error_section.py,
quant/weekly_audit_clip_rate_acceptance.py). Those bound quant error or
drift; this one bounds cross-precision disagreement.

Gamma variants (JuniorHome docs/WINSOR.md):
  absmean  BitNet absmean, mean |W|
  junior   mean |W| after dropping the top 1/16 by magnitude
  p95      95th percentile of |W| (nearest rank), the Winsor clip

fp32 is emulated exactly with struct round-trips: each weight is rounded to
fp32 and the mean is a sequential fp32 sum (each add rounded), divided in
fp32. With u = 2^-24 and k summed terms, the gamma bound is
  |g32 - g64| <= g64 * ((k + 2) * u) / (1 - (k + 2) * u)      (absmean, junior)
  |g32 - g64| <= g64 * u                                       (p95, one rounding)
A code q = clamp(round(w / g), -1, 1) may differ between precisions only if
| |w| / g64 - 0.5 | <= band, band = 0.5 * d + 2u, d = |g32 - g64| / g64
(the threshold 1.5 never changes the code because clamp maps both sides to +-1).
Winsor rate check: p95 clip fraction (|w| > g) must be <= 0.05 at both
precisions.

Trit per (engine, port, variant): +1 bound, flips and rate all hold, 0 no
sample, -1 a break. Folded per engine/port, then per day: receipt =
sha256(prev || pack || text)[:16], i2s 2-bit pack (0->0, 1->1, -1->2), MSB
first, zero-padded; matches the audit ledger vectors (pack 4a44, commit
e038848e2422d22a). Rate 0.05 (JuniorHome docs/JUNIORS_RATE.md); rate lock
sha256(prev || pack || b"0.05").

Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM. Bind 127.0.0.1
(stated only; no socket is opened). Offline: no subprocess, no file reads,
no network, no model pull, no GGUF download, no UE5 launch.
mint false, socket false, layer_one false, fetch false, ue5_launch false.

Run: python3 daily_audit_1645_fp32_gamma_agreement.py --selftest
     python3 daily_audit_1645_fp32_gamma_agreement.py --run 2026-10-23
     (stdin, optional: JSON {"BitNet-mlx": {"Home": {"absmean": g, ...}}, ...}
      replaces the emulated fp32 gamma with an engine-reported one)
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys

RATE = 0.05
RATE_BYTES = b"0.05"
GENESIS = "0" * 16
ROADMAP_ITEM = "JuniorHome docs/ROADMAP.md Automations ON: 16:45 daily audit"
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
MODEL_PULL = False
BIND = "127.0.0.1"
I2SD = "127.0.0.1:8767"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("BitNet-mlx", "JuniorQuant")
COMPATIBLE_ENGINES = ("BitNet-mlx", "JuniorQuant", "JuniorStock", "stocksnode",
                      "web3node", "JuniorLLM")
CLOSE = "sha256(prev || pack || text)[:16]"
VARIANTS = ("absmean", "junior", "p95")
SAMPLE_N = 1536
U32 = 2.0 ** -24

_ENC = {0: 0, 1: 1, -1: 2}
_VEC_TRITS = (1, 0, -1, -1, 1, 0, 1)
_VEC_PACK = "4a44"
_VEC_COMMIT = "e038848e2422d22a"


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


def side_commit(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def rate_commit(prev: str, pk: bytes) -> str:
    return hashlib.sha256(prev.encode() + pk + RATE_BYTES).hexdigest()


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def fixture(engine: str, port: str, day: str, n: int = SAMPLE_N) -> list:
    """Deterministic heavy-tailed weights: splitmix64, Gaussian scale mixture."""
    s = int.from_bytes(hashlib.sha256(f"fp32|{engine}|{port}|{day}".encode()).digest()[:8], "big")
    m = 0xFFFFFFFFFFFFFFFF

    def nxt() -> float:
        nonlocal s
        s = (s + 0x9E3779B97F4A7C15) & m
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
        z ^= z >> 31
        return ((z >> 11) + 0.5) / float(1 << 53)

    out = []
    for _ in range(n):
        g = math.sqrt(-2.0 * math.log(nxt())) * math.cos(2.0 * math.pi * nxt())
        scale = 0.06 if nxt() < 0.03 else 0.015
        out.append(g * scale)
    return out


def _sorted_abs(w) -> list:
    return sorted(abs(x) for x in w)


def _keep_n(n: int) -> int:
    return n - n // 16 if n >= 16 else n


def gammas64(w) -> dict:
    a = _sorted_abs(w)
    n = len(a)
    k = _keep_n(n)
    rank = max(1, math.ceil(0.95 * n))
    return {"absmean": math.fsum(a) / n, "junior": math.fsum(a[:k]) / k, "p95": a[rank - 1]}


def gammas32(w) -> dict:
    a = sorted(abs(f32(x)) for x in w)
    n = len(a)
    k = _keep_n(n)
    rank = max(1, math.ceil(0.95 * n))

    def mean32(xs) -> float:
        acc = 0.0
        for x in xs:
            acc = f32(acc + x)
        return f32(acc / f32(float(len(xs))))

    return {"absmean": mean32(a), "junior": mean32(a[:k]), "p95": a[rank - 1]}


def gamma_bound(variant: str, g64: float, n: int) -> float:
    if variant == "p95":
        return g64 * U32
    k = n if variant == "absmean" else _keep_n(n)
    t = (k + 2) * U32
    return g64 * t / (1.0 - t)


def code(x: float, g: float) -> int:
    return max(-1, min(1, round(x / g)))


def flip_explained(x: float, g64: float, d: float) -> bool:
    """A code flip is rounding only if |w| / g64 is within the band of 0.5."""
    return abs(abs(x) / g64 - 0.5) <= 0.5 * d + 2.0 * U32


def compare(variant: str, w, g64: float, g32: float) -> dict:
    n = len(w)
    d = abs(g32 - g64) / g64
    flips = unexplained = 0
    clip64 = clip32 = 0
    for x in w:
        x32 = f32(x)
        q64, q32 = code(x, g64), code(x32, g32)
        if q64 != q32:
            flips += 1
            if not flip_explained(x, g64, d):
                unexplained += 1
        clip64 += abs(x) > g64
        clip32 += abs(x32) > g32
    return {"g64": g64, "g32": g32, "diff": abs(g32 - g64),
            "bound": gamma_bound(variant, g64, n), "flips": flips,
            "unexplained": unexplained, "clip64": clip64 / n, "clip32": clip32 / n}


def row_trit(variant: str, c: dict | None) -> int:
    if c is None:
        return 0
    ok = c["diff"] <= c["bound"] and c["unexplained"] == 0
    if variant == "p95":
        ok = ok and c["clip64"] <= RATE + 1e-12 and c["clip32"] <= RATE + 1e-12
    return 1 if ok else -1


def day_section(day: str, reported: dict | None = None, empty=(), prev: str = GENESIS) -> dict:
    reported = reported or {}
    lines = [f"## fp32 gamma agreement (16:45) {day}", "",
             "| engine | port | gamma | fp64 | fp32 | diff | bound | flips | unexpl | trit |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    trits, labels, tip = [], [], prev
    for e in ENGINES:
        for p in PORTS:
            w = [] if (e, p) in empty else fixture(e, p, day)
            pt = []
            if w:
                g64s, g32s = gammas64(w), gammas32(w)
                g32s.update(reported.get(e, {}).get(p, {}))
            for v in VARIANTS:
                c = compare(v, w, g64s[v], float(g32s[v])) if w else None
                t = row_trit(v, c)
                pt.append(t)
                labels.append(f"{e}/{p}/{v}")
                if c is None:
                    lines.append(f"| {e} | {p} | {v} | - | - | - | - | - | - | 0 |")
                else:
                    lines.append(f"| {e} | {p} | {v} | {c['g64']:.9f} | {c['g32']:.9f} | "
                                 f"{c['diff']:.2e} | {c['bound']:.2e} | {c['flips']} | "
                                 f"{c['unexplained']} | {t:+d} |")
            trits.extend(pt)
            tip = side_commit(tip, pack(pt), f"{day}|{e}|{p}")
    pk = pack(trits)
    receipt = side_commit(prev, pk, f"{day}|{tip}")
    lines += ["", f"receipt {receipt} prev {prev} rate {RATE} bind {BIND} i2sd {I2SD}"]
    return {"day": day, "trits": trits, "pack": pk.hex(), "tip": tip, "receipt": receipt,
            "rate_lock": rate_commit(prev, pk),
            "breaks": [lab for lab, t in zip(labels, trits) if t == -1],
            "markdown": "\n".join(lines) + "\n"}


def selftest() -> None:
    assert pack(_VEC_TRITS).hex() == _VEC_PACK
    assert side_commit(GENESIS, pack(_VEC_TRITS), "stocks") == _VEC_COMMIT
    assert pack([]) == b"" and pack([1, 1, 1, 1]).hex() == "55"
    assert f32(0.1) != 0.1 and f32(f32(0.1)) == f32(0.1) and f32(1.5) == 1.5
    # gammas on an exact vector agree bit-for-bit in both precisions
    w = [float(i) for i in range(1, 21)]
    g64, g32 = gammas64(w), gammas32(w)
    assert g64 == {"absmean": 10.5, "junior": 10.0, "p95": 19.0}
    assert g32 == g64
    # every engine/port/variant passes on the deterministic fixture
    a = day_section("2026-10-23")
    assert a["breaks"] == [], a["breaks"]
    assert len(a["trits"]) == len(ENGINES) * len(PORTS) * len(VARIANTS) == 30
    assert all(t == 1 for t in a["trits"])
    # fp32 actually differs from fp64 (the check is not vacuous) and stays bounded
    wf = fixture("BitNet-mlx", "Home", "2026-10-23")
    gg64, gg32 = gammas64(wf), gammas32(wf)
    assert any(gg64[v] != gg32[v] for v in VARIANTS)
    for v in VARIANTS:
        c = compare(v, wf, gg64[v], gg32[v])
        assert c["diff"] <= c["bound"] and c["unexplained"] == 0
    assert compare("p95", wf, gg64["p95"], gg32["p95"])["clip64"] <= RATE
    # a reported gamma off by 1e-4 relative breaks the bound -> -1 and a new receipt
    bad = {"JuniorQuant": {"OSai": {"absmean": gg64["absmean"] * 1.0001 + 1e-3}}}
    b = day_section("2026-10-23", reported=bad)
    assert "JuniorQuant/OSai/absmean" in b["breaks"] and b["receipt"] != a["receipt"]
    # flips from a shifted gamma land in the band; a flip far from 0.5 is not rounding
    c = compare("absmean", [0.46, -0.47, 0.2, 1.0], 1.0, 0.9)
    assert c["flips"] == 2 and c["unexplained"] == 0 and row_trit("absmean", c) == -1
    assert flip_explained(0.4999999, 1.0, 1e-6) and not flip_explained(0.3, 1.0, 1e-6)
    assert row_trit("junior", {"diff": 0.0, "bound": 1.0, "unexplained": 1,
                               "clip64": 0.0, "clip32": 0.0}) == -1
    # p95 clip-rate break and missing sample
    assert row_trit("p95", {"diff": 0.0, "bound": 1.0, "unexplained": 0,
                            "clip64": 0.2, "clip32": 0.0}) == -1
    z = day_section("2026-10-23", empty={("BitNet-mlx", "FieldCore")})
    assert z["trits"][6:9] == [0, 0, 0] and z["receipt"] != a["receipt"]
    # determinism, chaining and close rule recomputed from its parts
    assert day_section("2026-10-23") == a and len(a["receipt"]) == 16
    nxt = day_section("2026-10-24", prev=a["receipt"])
    assert nxt["receipt"] != a["receipt"] and f"prev {a['receipt']}" in nxt["markdown"]
    tip, i = GENESIS, 0
    for e in ENGINES:
        for p in PORTS:
            tip = side_commit(tip, pack(a["trits"][i:i + 3]), f"2026-10-23|{e}|{p}")
            i += 3
    assert tip == a["tip"]
    assert side_commit(GENESIS, bytes.fromhex(a["pack"]), f"2026-10-23|{tip}") == a["receipt"]
    assert a["rate_lock"] == rate_commit(GENESIS, bytes.fromhex(a["pack"]))
    assert a["markdown"].count("\n| ") == 1 + 30
    assert BIND == "127.0.0.1" and not (MINT or SOCKET or LAYER_ONE or FETCH
                                        or UE5_LAUNCH or MODEL_PULL)
    print(f"selftest ok receipt {a['receipt']} pack {a['pack']}")


def main(argv) -> int:
    if "--selftest" in argv:
        selftest()
        return 0
    if "--run" in argv:
        i = argv.index("--run")
        day = argv[i + 1] if len(argv) > i + 1 else "1970-01-01"
        reported = None
        if not sys.stdin.isatty():
            raw = sys.stdin.read().strip()
            reported = json.loads(raw) if raw else None
        out = day_section(day, reported)
        sys.stdout.write(out["markdown"])
        return 1 if out["breaks"] else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
