"""KV-cache Winsor clip and attention error bound under bitnetd.service MemoryMax.

Stdlib only, offline. Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
TCP port named, never dialed: bitnetd 127.0.0.1:8765 (i2sd 127.0.0.1:8767 too).

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay": hardened
bitnetd.service bound to 127.0.0.1:8765 with MemoryMax=2G (JuniorLLM
rails/linux/bitnetd.service). Quant Chief, pair A, slot 2026-10-18 10:20 MT.
This is the quant-error / Winsor-bound side only. It writes no service file and
no OS config. Unit hardening is covered by web3node 5dda3d5, the MemoryMax
streaming gamma for weights by JuniorStock quant/calibration/
bitnetd_memcap_stream_gamma.py, and MemoryMax vs I2_S pack size by Trit Pack &
I2_S (BitNet-mlx scripts/bitnetd_memmax_i2s_pack_key.py). This file takes the
pack size as an input and does not re-derive it.

The gap: the ternary weights are only part of what bitnetd holds under the cap.
The K/V cache grows with context, and the bytes left after the I2_S pack decide
how many bits each cached K/V entry may keep. Fewer bits need a clip (Winsor)
threshold, and nothing so far says what that costs at the attention output.

New capability, per bit width b in BITS and per Winsor clip c in CLIPS:
  memory  headroom = MemoryMax - pack_bytes - overhead
          bytes/token = 2 * layers * kv_heads * head_dim * b / 8
          max_ctx = headroom // bytes/token, fits = max_ctx >= ctx
  quant   symmetric int-b, qmax = 2^(b-1) - 1, s = c / qmax,
          Q(x) = s * clamp(floor(x/s + 1/2), -qmax, qmax)
          |x - Q(x)| <= s/2 when |x| <= c, else |x| - c + 0 (clip excess)
  logits  sample:   eps = max_q max_t |q.(k_t - Q(k_t))| / sqrt(d)  (exact)
          in range: eps = max_q ||q||_1 * (s/2) / sqrt(d)   (Hoelder, any
          token whose K entries lie within +-c)
  softmax every logit moves by <= eps, so p'_t / p_t lies in
          [e^-2eps, e^2eps] and ||p' - p||_1 <= e^(2eps) - 1
  output  ||o - o'||_2 <= (e^(2eps) - 1) * max ||v_t|| + max ||v_t - Q(v_t)||
          rel = that bound / max ||v_t||, compared with RATE = 0.05
Two versions of rel: rel_sample uses the exact errors of the calibration cache
(certified for that cache), rel_inrange uses s/2 per coordinate (certified for
any later token whose K/V entries lie within +-c, for the calibration queries). Clip candidates are the
Winsor nearest-rank percentiles of |x|: amax (no clip), P99.9, P99, P95.
Verdict trit per (b, clip): +1 fits and rel_inrange <= RATE and clipped share
<= RATE; 0 fits and rel_sample <= RATE only; -1 does not fit or rel_sample >
RATE. The plan picks, for each b, the clip with the smallest rel_sample, and
recommends the largest b that fits with verdict >= 0.

Close rule: sha256(prev || pack || text)[:16], GENESIS prev = "0" * 16. Per b
row: pack = I2_S 2-bit code of its verdict trit (0->0, 1->1, -1->2, MSB first),
text = canonical JSON of the row (floats as float.hex). Rows chain in BITS
order, then the report key: pack = I2_S code of all row verdicts, text =
canonical JSON of the report. Rate 0.05 is the attention-output error budget.
Compatible engines: bitnetd 127.0.0.1:8765, BitNet-mlx, JuniorLLM, JuniorQuant,
i2sd 127.0.0.1:8767 (named, never dialed), JuniorOS overlay (Debian/Alpine).
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no subprocess, no team import.

Default geometry is an assumption for BitNet b1.58 2B4T (30 layers, 5 KV heads,
head_dim 128, ctx 4096); override with --geometry L,H,D and --ctx.

Usage:
  python3 bitnetd_kv_memmax_winsor_bound.py --selftest
  python3 bitnetd_kv_memmax_winsor_bound.py --pack-bytes 1200000000 \
      [--memory-max 2G] [--geometry 30,5,128] [--ctx 4096] [--seed 7]
"""
from __future__ import annotations

import argparse
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
BITNETD_PORT = 8765
I2SD_PORT = 8767
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
GENESIS = "0" * 16
UNIT_MEMORY_MAX = "2G"          # JuniorLLM rails/linux/bitnetd.service
DEFAULT_OVERHEAD = 256 << 20    # runtime + activations (assumption)
DEFAULT_GEOMETRY = (30, 5, 128) # layers, kv_heads, head_dim (2B4T, assumption)
DEFAULT_CTX = 4096
BITS = (8, 6, 4, 3)
CLIPS = ("amax", "p999", "p99", "p95")
_PCT = {"p999": 0.999, "p99": 0.99, "p95": 0.95}
SLACK = 1e-9                    # relative fp slack added to every bound
_ENC = {0: 0, 1: 1, -1: 2}


def pack_i2s(trits) -> bytes:
    acc = bits = 0
    out = bytearray()
    for t in trits:
        acc = (acc << 2) | _ENC[t]
        bits += 2
        if bits == 8:
            out.append(acc)
            acc = bits = 0
    if bits:
        out.append(acc << (8 - bits))
    return bytes(out)


def close_rule(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def memory_bytes(s: str) -> int:
    """systemd size suffixes, base 1024."""
    s = s.strip().upper()
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    if s and s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(s)


# ---- memory side -----------------------------------------------------------
def kv_bytes_per_token(geometry, b: int) -> float:
    layers, kv_heads, head_dim = geometry
    return 2 * layers * kv_heads * head_dim * b / 8


def max_ctx(memory_max: str, pack_bytes: int, overhead: int, geometry, b: int) -> int:
    head = memory_bytes(memory_max) - pack_bytes - overhead
    if head <= 0:
        return 0
    return int(head // kv_bytes_per_token(geometry, b))


# ---- quantizer -------------------------------------------------------------
def clip_value(values, which: str) -> float:
    a = sorted(abs(x) for x in values)
    if not a:
        return 0.0
    if which == "amax":
        return a[-1]
    r = max(1, math.ceil(_PCT[which] * len(a)))
    return a[r - 1]


def quant(x: float, c: float, b: int) -> float:
    if c <= 0:
        return 0.0
    qmax = (1 << (b - 1)) - 1
    s = c / qmax
    k = math.floor(x / s + 0.5)
    k = qmax if k > qmax else (-qmax if k < -qmax else k)
    return k * s


def _norm(v) -> float:
    return math.sqrt(math.fsum(x * x for x in v))


def _dot(a, b) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def certify(keys, vals, queries, b: int, clip: str) -> dict:
    """Bounds for one bit width and one Winsor clip on a calibration cache."""
    d = len(keys[0])
    flat_k = [x for row in keys for x in row]
    flat_v = [x for row in vals for x in row]
    ck, cv = clip_value(flat_k, clip), clip_value(flat_v, clip)
    qmax = (1 << (b - 1)) - 1
    sk, sv = (ck / qmax if ck else 0.0), (cv / qmax if cv else 0.0)
    clipped = sum(1 for x in flat_k if abs(x) > ck) + sum(1 for x in flat_v if abs(x) > cv)
    errk = [[x - quant(x, ck, b) for x in row] for row in keys]
    ev = max(_norm([x - quant(x, cv, b) for x in row]) for row in vals)
    rd = math.sqrt(d)
    eps_s = max(abs(_dot(q, e)) for q in queries for e in errk) / rd * (1 + SLACK)
    q1 = max(math.fsum(abs(x) for x in q) for q in queries)
    eps_i = q1 * (sk / 2) / rd * (1 + SLACK)
    vmax = max(_norm(v) for v in vals)

    def out_bound(eps, e_v):
        return (math.expm1(2 * eps) * vmax + e_v) * (1 + SLACK)

    ob_s = out_bound(eps_s, ev)
    ob_i = out_bound(eps_i, rd * sv / 2)
    ob_i = max(ob_i, ob_s) if clipped == 0 else ob_i
    rel_s = ob_s / vmax if vmax > 0 else 0.0
    rel_i = ob_i / vmax if vmax > 0 else 0.0
    share = clipped / (len(flat_k) + len(flat_v))
    return {"b": b, "clip": clip, "ck": ck, "cv": cv, "eps_sample": eps_s,
            "eps_inrange": eps_i, "out_bound_sample": ob_s, "out_bound_inrange": ob_i,
            "rel_sample": rel_s, "rel_inrange": rel_i, "clipped_share": share}


def verdict(row: dict, fits: bool) -> int:
    if not fits or row["rel_sample"] > RATE:
        return -1
    if row["rel_inrange"] <= RATE and row["clipped_share"] <= RATE:
        return 1
    return 0


def plan(keys, vals, queries, pack_bytes: int, memory_max: str = UNIT_MEMORY_MAX,
         overhead: int = DEFAULT_OVERHEAD, geometry=DEFAULT_GEOMETRY,
         ctx: int = DEFAULT_CTX, prev: str = GENESIS) -> dict:
    rows = []
    key = prev
    for b in BITS:
        cands = [certify(keys, vals, queries, b, c) for c in CLIPS]
        best = min(cands, key=lambda r: (r["rel_sample"], CLIPS.index(r["clip"])))
        mc = max_ctx(memory_max, pack_bytes, overhead, geometry, b)
        fits = mc >= ctx
        row = dict(best)
        row.update({"max_ctx": mc, "fits": fits,
                    "bytes_per_token": kv_bytes_per_token(geometry, b),
                    "verdict": verdict(best, fits)})
        text = canon({k: (v.hex() if isinstance(v, float) else v) for k, v in row.items()})
        key = close_rule(key, pack_i2s([row["verdict"]]), text)
        row["key"] = key
        rows.append(row)
    ok = [r for r in rows if r["verdict"] >= 0]
    rec = max(ok, key=lambda r: r["b"])["b"] if ok else None
    verdicts = [r["verdict"] for r in rows]
    report = {
        "memory_max": memory_max, "pack_bytes": pack_bytes, "overhead": overhead,
        "geometry": list(geometry), "ctx": ctx, "rate": RATE, "recommend_bits": rec,
        "verdicts": verdicts,
        "rows": [{k: (v.hex() if isinstance(v, float) else v) for k, v in r.items()}
                 for r in rows],
        "bind": BIND, "bitnetd_port": BITNETD_PORT, "i2sd_port": I2SD_PORT,
        "ports": list(PORTS), "mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
        "fetch": FETCH, "ue5_launch": UE5_LAUNCH,
    }
    report["key"] = close_rule(key, pack_i2s(verdicts), canon(report))
    return report


# ---- synthetic calibration cache (Irwin-Hall, random.random only) ----------
def synthetic(t: int, d: int, nq: int, seed: int, outliers: bool = True):
    rng = random.Random(seed)

    def ih():
        return rng.random() + rng.random() + rng.random() + rng.random() - 2.0

    out_ch = {3, d // 2} if outliers else set()
    keys = [[ih() * (6.0 if j in out_ch else 1.0) for j in range(d)] for _ in range(t)]
    vals = [[ih() * 0.5 for _ in range(d)] for _ in range(t)]
    if outliers:
        for i in range(0, t, 17):
            vals[i][1] *= 9.0
    queries = [[ih() * 0.3 for _ in range(d)] for _ in range(nq)]
    return keys, vals, queries


def attention(q, keys, vals):
    rd = math.sqrt(len(q))
    z = [_dot(q, k) / rd for k in keys]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    tot = math.fsum(e)
    p = [x / tot for x in e]
    return [math.fsum(pi * v[j] for pi, v in zip(p, vals)) for j in range(len(vals[0]))], z, p


def selftest() -> int:
    fails = 0

    def ok(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            fails += 1

    ok("flags false, loopback, five ports",
       not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
       and BIND == "127.0.0.1" and len(PORTS) == 5 and BITNETD_PORT == 8765)
    ok("MemoryMax 2G is base 1024", memory_bytes("2G") == 2 * 1024 ** 3)
    ok("2B4T KV bytes/token at 8 bits", kv_bytes_per_token(DEFAULT_GEOMETRY, 8) == 38400)
    pack = 1_200_000_000
    mcs = [max_ctx("2G", pack, DEFAULT_OVERHEAD, DEFAULT_GEOMETRY, b) for b in BITS]
    ok(f"max_ctx grows as bits shrink {mcs}", all(x < y for x, y in zip(mcs, mcs[1:])))
    ok("no headroom gives max_ctx 0", max_ctx("1G", 2 << 30, 0, DEFAULT_GEOMETRY, 4) == 0)
    ok("pack_i2s codes", pack_i2s([1, -1, 0, 1]) == bytes([0b01100001]))
    ok("quant in range error <= s/2",
       all(abs(x - quant(x, 1.0, 4)) <= (1.0 / 7) / 2 * (1 + 1e-12)
           for x in [i / 997 - 1.0 for i in range(1995)]))
    ok("quant clamps at c", quant(5.0, 1.0, 4) == 1.0 and quant(-5.0, 1.0, 4) == -1.0)

    keys, vals, queries = synthetic(64, 32, 12, 7)
    worst = {}
    for b in BITS:
        for c in CLIPS:
            r = certify(keys, vals, queries, b, c)
            qk = [[quant(x, r["ck"], b) for x in row] for row in keys]
            qv = [[quant(x, r["cv"], b) for x in row] for row in vals]
            eps_ok = out_ok = l1_ok = True
            for q in queries:
                o, z, p = attention(q, keys, vals)
                o2, z2, p2 = attention(q, qk, qv)
                if max(abs(a - a2) for a, a2 in zip(z, z2)) > r["eps_sample"] * (1 + SLACK):
                    eps_ok = False
                if math.fsum(abs(a - a2) for a, a2 in zip(p, p2)) > \
                        math.expm1(2 * r["eps_sample"]) * (1 + SLACK) + 1e-15:
                    l1_ok = False
                if _norm([a - a2 for a, a2 in zip(o, o2)]) > r["out_bound_sample"]:
                    out_ok = False
            ok(f"b={b} {c} logit error <= eps", eps_ok)
            ok(f"b={b} {c} softmax L1 <= e^2eps - 1", l1_ok)
            ok(f"b={b} {c} output error <= certified", out_ok)
            if c == "amax":
                ok(f"b={b} amax clips nothing, inrange >= sample",
                   r["clipped_share"] == 0 and r["rel_inrange"] >= r["rel_sample"])
            worst[(b, c)] = r["rel_inrange"]
    ok("in-range bound grows as bits shrink",
       all(worst[(x, "amax")] < worst[(y, "amax")] for x, y in zip(BITS, BITS[1:])))
    ok("tighter clip gives tighter in-range bound at b=4",
       worst[(4, "p95")] < worst[(4, "amax")])

    rep = plan(keys, vals, queries, pack, ctx=4096)
    ok(f"plan verdicts {rep['verdicts']} recommend {rep['recommend_bits']}",
       len(rep["verdicts"]) == len(BITS) and all(v in (-1, 0, 1) for v in rep["verdicts"]))
    ok("outlier cache: 8-bit row not red", rep["verdicts"][0] >= 0)
    ok("outlier cache: 3-bit row red", rep["verdicts"][-1] == -1)
    mk, mv, mq = synthetic(64, 32, 12, 7, outliers=False)
    mild = plan(mk, mv, mq, pack, ctx=4096)
    ok(f"mild cache: 8-bit row green {mild['verdicts']}", mild["verdicts"][0] == 1)
    ok("recommendation is the widest non-red width",
       rep["recommend_bits"] == max(b for b, v in zip(BITS, rep["verdicts"]) if v >= 0))
    tight = plan(keys, vals, queries, memory_bytes("2G") - DEFAULT_OVERHEAD - 50_000_000,
                 ctx=4096)
    ok(f"tight cap fails wide widths {tight['verdicts']}", tight["verdicts"][0] == -1
       and tight["rows"][0]["fits"] is False)
    none = plan(keys, vals, queries, 3 << 30, ctx=1)
    ok("pack over MemoryMax: all red, no recommendation",
       none["verdicts"] == [-1] * len(BITS) and none["recommend_bits"] is None)
    r1 = plan(keys, vals, queries, pack)
    r2 = plan(keys, vals, queries, pack)
    r3 = plan(keys, vals, queries, pack, prev="f" * 16)
    ok("key deterministic, 16 hex", r1["key"] == r2["key"] and len(r1["key"]) == 16
       and int(r1["key"], 16) >= 0)
    ok("key depends on prev", r1["key"] != r3["key"])
    ok("report JSON canonical", json.loads(canon(r1))["key"] == r1["key"])
    zero = [[0.0] * 8 for _ in range(4)]
    rz = certify(zero, zero, [[1.0] * 8], 4, "p95")
    ok("all-zero cache gives zero bound", rz["out_bound_sample"] == 0.0
       and verdict(rz, True) == 1)

    print(f"selftest: {'PASS' if fails == 0 else 'FAIL'} ({fails} failed)")
    return 0 if fails == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pack-bytes", type=int)
    ap.add_argument("--memory-max", default=UNIT_MEMORY_MAX)
    ap.add_argument("--overhead", type=int, default=DEFAULT_OVERHEAD)
    ap.add_argument("--geometry", default=",".join(str(x) for x in DEFAULT_GEOMETRY))
    ap.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.pack_bytes is None:
        ap.print_help()
        return 2
    geom = tuple(int(x) for x in a.geometry.split(","))
    keys, vals, queries = synthetic(64, 32, 12, a.seed)
    rep = plan(keys, vals, queries, a.pack_bytes, a.memory_max, a.overhead, geom,
               a.ctx, a.prev)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if rep["recommend_bits"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
