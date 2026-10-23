"""16:45 daily audit: quant-error section for the remaining engines.

Roadmap item: JuniorHome docs/ROADMAP.md, Automations (current) ON:
"16:45 daily audit", named for pair A by Quant Chief on 2026-10-22 18:10 MT
("remaining engines only"). The audit already has a receipt ledger
(stocksnode 4eb7e33), a JuniorLLM side (f483bf0) and a JuniorHome slot/port
map (062c856). None of them put a quant-error number in the audit. This file
adds that: one "## Quant error (16:45)" markdown section per day, one row per
engine port and Winsor gamma variant, so the audit records whether the
ternary quantization error the i2sd userspace (127.0.0.1:8767) and BitNet
ternary inference rely on stayed inside its analytic bound that day.

New capability. It never imports, runs, wraps, tests, rewrites or re-derives
any team file or any earlier Winsor & Quant Error file (JuniorQuant
src/winsor_clip.py, src/i2sd_quant_budget.py, src/i2sd_residual_bound.py,
src/i2sd_act_accum_bound.py; JuniorStock risk/winsor_quant_error.py,
security/winsor_outlier_health.py, quant/calibration/winsor_night_drift.py).
Those compute gammas or budgets for one tensor; this one emits the audit's
per-engine quant-error rows with a per-element bound check and a chained
receipt.

Gamma variants (JuniorHome docs/WINSOR.md):
  absmean  BitNet absmean, mean |W|
  junior   mean |W| after dropping the top 1/16 by magnitude
  p95      95th percentile of |W| (nearest rank), the Winsor clip

Per (engine, variant), with q = clamp(round(w / g), -1, 1), r = q * g:
  err   = sum (w - r)^2 / sum w^2                     (relative MSE)
  bound = sum of per-element caps / sum w^2, where
          in-range elements (|w| <= 1.5 g): per-element error <= (g/2)^2
          clipped elements  (|w| >  1.5 g): error = (|w| - g)^2
  clip  = fraction with |w| > g   (Winsor rate check: p95 must be <= 0.05)
Trit: +1 err <= bound and the rate check holds, 0 no sample, -1 a break.
Folded per engine, then per day: day receipt =
sha256(prev || pack || text)[:16], i2s 2-bit pack (0->0, 1->1, -1->2),
MSB first, zero-padded; matches the audit ledger vectors (pack 4a44,
commit e038848e2422d22a). Rate 0.05 (JuniorHome docs/JUNIORS_RATE.md);
rate lock sha256(prev || pack || b"0.05").

Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM. Bind 127.0.0.1
(stated only; no socket is opened). Offline: no subprocess, no file reads,
no network, no model pull, no GGUF download, no UE5 launch.
mint false, socket false, layer_one false, fetch false, ue5_launch false.

Run: python3 daily_audit_1645_quant_error_section.py --selftest
     python3 daily_audit_1645_quant_error_section.py --run 2026-10-22
     (stdin, optional: JSON {"Engine": [w, ...], ...} replaces the
      deterministic per-engine reference sample)
"""
from __future__ import annotations

import hashlib
import json
import math
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
COMPATIBLE_ENGINES = ("stocksnode", "web3node", "JuniorLLM", "BitNet-mlx")
CLOSE = "sha256(prev || pack || text)[:16]"
VARIANTS = ("absmean", "junior", "p95")
SAMPLE_N = 2048

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


def reference_sample(engine: str, day: str, n: int = SAMPLE_N) -> list:
    """Deterministic heavy-tailed weights (Laplace via inverse CDF, xorshift64)."""
    s = int.from_bytes(hashlib.sha256(f"{engine}|{day}".encode()).digest()[:8], "big") or 1
    out = []
    for _ in range(n):
        s ^= (s << 13) & 0xFFFFFFFFFFFFFFFF
        s ^= s >> 7
        s ^= (s << 17) & 0xFFFFFFFFFFFFFFFF
        u = ((s >> 11) + 0.5) / float(1 << 53) - 0.5
        out.append(-0.02 * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u)))
    return out


def gammas(w) -> dict:
    a = sorted(abs(x) for x in w)
    n = len(a)
    keep = a[: n - n // 16] if n >= 16 else a
    rank = max(1, math.ceil(0.95 * n))
    return {"absmean": sum(a) / n,
            "junior": sum(keep) / len(keep),
            "p95": a[rank - 1]}


def quant_error(w, g: float) -> dict:
    den = sum(x * x for x in w)
    err = bnd = 0.0
    clipped = 0
    for x in w:
        q = max(-1, min(1, round(x / g)))
        err += (x - q * g) ** 2
        ax = abs(x)
        bnd += (ax - g) ** 2 if ax > 1.5 * g else (g / 2.0) ** 2
        clipped += ax > g
    return {"err": err / den, "bound": bnd / den, "clip": clipped / len(w)}


def row_trit(variant: str, qe: dict | None) -> int:
    if qe is None:
        return 0
    ok = qe["err"] <= qe["bound"] + 1e-12
    if variant == "p95":
        ok = ok and qe["clip"] <= RATE + 1e-12
    return 1 if ok else -1


def engine_rows(engine: str, w) -> list:
    if not w:
        return [(engine, v, None, None, 0) for v in VARIANTS]
    gs = gammas(w)
    rows = []
    for v in VARIANTS:
        qe = quant_error(w, gs[v])
        rows.append((engine, v, gs[v], qe, row_trit(v, qe)))
    return rows


def day_section(day: str, samples: dict | None = None, prev: str = GENESIS) -> dict:
    samples = samples or {}
    lines = [f"## Quant error (16:45) {day}", "",
             "| port | gamma | value | rel err | bound | clip | trit |",
             "|---|---|---|---|---|---|---|"]
    trits, engine_tips, tip = [], {}, prev
    for e in PORTS:
        w = samples[e] if e in samples else reference_sample(e, day)
        rows = engine_rows(e, w)
        et = [r[4] for r in rows]
        trits.extend(et)
        for (_, v, g, qe, t) in rows:
            if qe is None:
                lines.append(f"| {e} | {v} | - | - | - | - | 0 |")
            else:
                lines.append(f"| {e} | {v} | {g:.6f} | {qe['err']:.4f} | "
                             f"{qe['bound']:.4f} | {qe['clip']:.4f} | {t:+d} |")
        tip = side_commit(tip, pack(et), f"{day}|{e}")
        engine_tips[e] = tip
    pk = pack(trits)
    receipt = side_commit(prev, pk, f"{day}|{tip}")
    lines += ["", f"receipt {receipt} prev {prev} rate {RATE} bind {BIND}"]
    return {"day": day, "trits": trits, "pack": pk.hex(), "engine_tips": engine_tips,
            "receipt": receipt, "rate_lock": rate_commit(prev, pk),
            "breaks": [f"{r}" for r, t in zip(
                [f"{e}/{v}" for e in PORTS for v in VARIANTS], trits) if t == -1],
            "markdown": "\n".join(lines) + "\n"}


def selftest() -> None:
    assert pack(_VEC_TRITS).hex() == _VEC_PACK
    assert side_commit(GENESIS, pack(_VEC_TRITS), "stocks") == _VEC_COMMIT
    assert pack([]) == b"" and pack([1, 1, 1, 1]).hex() == "55"
    # gammas on a known vector
    w = [float(i) for i in range(1, 21)]  # |W| = 1..20
    g = gammas(w)
    assert abs(g["absmean"] - 10.5) < 1e-12
    assert abs(g["junior"] - sum(range(1, 20)) / 19) < 1e-12  # drop top 1 (20//16)
    assert g["p95"] == 19.0
    # bound holds on every variant, every engine, deterministic sample
    for e in PORTS:
        for (_, v, gv, qe, t) in engine_rows(e, reference_sample(e, "2026-10-22")):
            assert qe["err"] <= qe["bound"] + 1e-12, (e, v)
            assert 0.0 < qe["err"] < 1.0
            assert t == 1, (e, v, qe)
            if v == "p95":
                assert qe["clip"] <= RATE
    # absmean on Laplace: E|W| = b, so gamma near 0.02
    ga = gammas(reference_sample("Home", "2026-10-22"))
    assert 0.017 < ga["absmean"] < 0.023 and ga["junior"] < ga["absmean"] < ga["p95"]
    # determinism and chaining
    a = day_section("2026-10-22")
    b = day_section("2026-10-22")
    assert a == b and len(a["receipt"]) == 16 and len(a["trits"]) == 15
    c = day_section("2026-10-23", prev=a["receipt"])
    assert c["receipt"] != a["receipt"] and f"prev {a['receipt']}" in c["markdown"]
    assert a["markdown"].count("\n| ") == 1 + 15
    # missing sample -> 0 trits, break -> -1
    z = day_section("2026-10-22", {"OSai": []})
    assert z["trits"][3:6] == [0, 0, 0] and z["receipt"] != a["receipt"]
    assert row_trit("p95", {"err": 0.1, "bound": 0.2, "clip": 0.2}) == -1
    assert row_trit("absmean", {"err": 0.3, "bound": 0.2, "clip": 0.0}) == -1
    assert row_trit("junior", None) == 0
    # receipt matches the close rule recomputed from its parts
    tip = GENESIS
    for i, e in enumerate(PORTS):
        tip = side_commit(tip, pack(a["trits"][3 * i:3 * i + 3]), f"2026-10-22|{e}")
    assert side_commit(GENESIS, bytes.fromhex(a["pack"]), f"2026-10-22|{tip}") == a["receipt"]
    assert a["rate_lock"] == rate_commit(GENESIS, bytes.fromhex(a["pack"]))
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
        samples = None
        if not sys.stdin.isatty():
            raw = sys.stdin.read().strip()
            samples = json.loads(raw) if raw else None
        out = day_section(day, samples)
        sys.stdout.write(out["markdown"])
        return 1 if out["breaks"] else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
