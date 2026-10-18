"""Streaming Winsor gamma bracket under the hardened bitnetd.service MemoryMax.

Stdlib only, offline. Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay": Debian/Alpine +
os-release.junior + hardened bitnetd.service + juniorctl on PATH. This file is
the quant-error / Winsor-bound side of the hardened bitnetd.service part only
(Quant Chief, pair A, slot 2026-10-18 02:20 MT). It writes no service file and
no OS config, and it does not cover unit hardening itself (JuniorHome web3node
bitnetd_unit_hardening_receipt.py already does). The I2_S key side belongs to
Trit Pack & I2_S.

The gap: the hardened unit (JuniorLLM rails/linux/bitnetd.service) sets
MemoryMax=2G. Two of the three Winsor gammas in JuniorHome docs/WINSOR.md need
order statistics of |W|:
  absmean = mean|W|                                   (BitNet b1.58, streamable)
  drop    = mean|W| after dropping the top 1/16       (Junior drop, needs a rank)
  p95     = mean of min(|W|, P95), nearest-rank       (P95 clip, needs a rank)
A full sort of one CPython list of floats costs about 32 bytes per weight, so a
single 2560x6912 projection already wants ~566 MB, far over a 1/16 calibration
share (128 MiB) of MemoryMax=2G. Under the cap, bitnetd must compute drop and p95
from a bounded-memory sketch instead, and nothing so far says how wrong that is.

New capability: an O(B)-memory, two-pass streaming histogram whose bins keep
(count, compensated sum, min, max). Bin index int(|w|*B/max) is monotone in |w|,
so every value in bin b is <= every value in bin b+1 and rank cuts land in one
bin. The sum of the m smallest of c values in a bin with total S, min lo, max hi
is bracketed by
  [max(m lo, S - (c-m) hi),  min(m hi, S - (c-m) lo, m S / c)]
which gives a certified bracket [gL, gH] that contains the exact sorted gamma.
p95 uses the exact identity sum min(|w|, q) = (sum of the r smallest) + (n-r) q,
with r = ceil(0.95 n) and q in [lo, hi] of its bin (q = lo if m = 1, hi if m = c).

From the bracket it certifies the quantization error (JuniorHome
docs/QUANT_ERROR.md, SSE = sum (w - gamma t)^2, t = RoundClip(w/gamma, -1, 1))
between quantizing with the exact gamma and with the streamed midpoint:
  at risk:  k = weights in bins touching [gL/2, gH/2] (only these can flip a trit)
  scale:    |dSSE| <= D * sum_b c_b * max corner |2|w| - 2g|   (D = gH - gL)
  flip:     |dSSE| <= k * gH * D          (since 2|w| and g both lie in [gL, gH])
Relative to signal energy sum w^2, compared with RATE = 0.05. Verdict trit per
gamma: +1 no weight at risk (trit stream provably equals the sort path) and
rel <= RATE, 0 some at risk but rel <= RATE, -1 rel > RATE (raise B or
the share). It also plans the smallest power-of-two B that keeps every
verdict >= 0, and whether a plain sort fits the share at all.

Close rule: sha256(prev || pack || text)[:16], GENESIS prev = "0" * 16. Per
gamma: pack = I2_S 2-bit code of its verdict trit (0->0, 1->1, -1->2, MSB
first), text = canonical JSON of the bracket (float.hex). Chained absmean ->
drop -> p95, then report key: pack = I2_S code of the 3 verdicts, text =
canonical JSON of the report. Rate 0.05 is the SSE drift budget.
Compatible engines: BitNet-mlx, JuniorLLM, JuniorQuant, bitnetd 127.0.0.1:8765,
i2sd 127.0.0.1:8767 (named, never dialed), JuniorOS overlay (Debian/Alpine).
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no team-module import.

Usage:
  python3 bitnetd_memcap_stream_gamma.py --selftest
  python3 bitnetd_memcap_stream_gamma.py --synthetic 8192 --seed 7 [--bins 256]
  python3 bitnetd_memcap_stream_gamma.py --weights FILE [--memory-max 2G]
  python3 bitnetd_memcap_stream_gamma.py --plan 2560x6912 [--memory-max 2G]
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
U = 2.0 ** -53
GAMMAS = ("absmean", "drop", "p95")
UNIT_MEMORY_MAX = "2G"      # JuniorLLM rails/linux/bitnetd.service
DEFAULT_SHARE = 1.0 / 16    # calibration scratch share of MemoryMax (assumption)
BIN_BYTES = 40              # count, sum, comp, min, max
SORT_ELEM_BYTES = 32        # CPython list slot (8) + float object (24)
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


def parse_memory_max(s: str) -> int:
    """systemd size suffixes are base 1024."""
    s = s.strip().upper()
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    if s and s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(s)


def budget_bytes(memory_max: str, share: float) -> int:
    return int(parse_memory_max(memory_max) * share)


def bins_for_budget(memory_max: str = UNIT_MEMORY_MAX, share: float = DEFAULT_SHARE) -> int:
    return max(1, budget_bytes(memory_max, share) // BIN_BYTES)


def sort_fits(n: int, memory_max: str = UNIT_MEMORY_MAX, share: float = DEFAULT_SHARE) -> bool:
    return n * SORT_ELEM_BYTES <= budget_bytes(memory_max, share)


class _Neumaier:
    __slots__ = ("s", "c")

    def __init__(self):
        self.s = 0.0
        self.c = 0.0

    def add(self, x: float) -> None:
        t = self.s + x
        if abs(self.s) >= abs(x):
            self.c += (self.s - t) + x
        else:
            self.c += (x - t) + self.s
        self.s = t

    def value(self) -> float:
        return self.s + self.c


def stream_sketch(source, bins: int) -> dict:
    """Two passes over source() (a callable returning a fresh iterable)."""
    n = 0
    amax = 0.0
    tot = _Neumaier()
    energy = _Neumaier()
    for x in source():
        a = abs(x)
        n += 1
        amax = a if a > amax else amax
        tot.add(a)
        energy.add(x * x)
    cnt = [0] * bins
    sums = [_Neumaier() for _ in range(bins)]
    mn = [math.inf] * bins
    mx = [-math.inf] * bins
    for x in source():
        a = abs(x)
        b = min(bins - 1, int(a * bins / amax)) if amax > 0 else 0
        cnt[b] += 1
        sums[b].add(a)
        if a < mn[b]:
            mn[b] = a
        if a > mx[b]:
            mx[b] = a
    seen = sum(cnt)
    if seen != n:
        raise ValueError("source changed between passes")
    return {"n": n, "amax": amax, "total": tot.value(), "energy": energy.value(),
            "bins": bins, "cnt": cnt, "sum": [s.value() for s in sums],
            "mn": mn, "mx": mx}


def _smallest_m(c: int, s: float, lo: float, hi: float, m: int):
    if m <= 0:
        return 0.0, 0.0
    if m >= c:
        return s, s
    low = max(m * lo, s - (c - m) * hi)
    up = min(m * hi, s - (c - m) * lo, m * s / c)
    return low, max(low, up)


def _cut(sk: dict, r: int):
    """Bin holding rank r (1-based); returns (b, below_sum, below_count)."""
    below_s = 0.0
    below_c = 0
    parts = []
    for b in range(sk["bins"]):
        c = sk["cnt"][b]
        if below_c + c >= r and c:
            return b, math.fsum(parts), below_c
        below_c += c
        parts.append(sk["sum"][b])
    raise ValueError("rank out of range")


def _slack(n: int) -> float:
    return 16 * U + (n * U) ** 2 * 4


def bracket(sk: dict, which: str):
    n = sk["n"]
    if n == 0:
        return 0.0, 0.0
    if which == "absmean" or (which == "drop" and n // 16 == 0):
        g = sk["total"] / n
        lo, hi = g, g
    elif which == "drop":
        r = n - n // 16
        b, below, bc = _cut(sk, r)
        l_m, u_m = _smallest_m(sk["cnt"][b], sk["sum"][b], sk["mn"][b], sk["mx"][b], r - bc)
        lo, hi = (below + l_m) / r, (below + u_m) / r
    elif which == "p95":
        r = max(1, math.ceil(0.95 * n))
        b, below, bc = _cut(sk, r)
        c, m = sk["cnt"][b], r - bc
        l_m, u_m = _smallest_m(c, sk["sum"][b], sk["mn"][b], sk["mx"][b], m)
        q_lo = sk["mx"][b] if m == c else sk["mn"][b]
        q_hi = sk["mn"][b] if m == 1 else sk["mx"][b]
        lo = (below + l_m + (n - r) * q_lo) / n
        hi = min((below + u_m + (n - r) * q_hi) / n, sk["total"] / n)
    else:
        raise ValueError(which)
    e = _slack(n)
    return lo * (1 - e), hi * (1 + e)


def certify(sk: dict, which: str) -> dict:
    gl, gh = bracket(sk, which)
    d = gh - gl
    ghat = 0.5 * (gl + gh)
    t_lo, t_hi = 0.5 * gl * (1 - 4 * U), 0.5 * gh * (1 + 4 * U)
    k = 0
    scale = []
    for b in range(sk["bins"]):
        c = sk["cnt"][b]
        if not c:
            continue
        lo, hi = sk["mn"][b], sk["mx"][b]
        if hi >= t_lo and lo <= t_hi:
            k += c
        if hi >= t_lo:
            h = max(abs(2 * hi - 2 * gl), abs(2 * lo - 2 * gh),
                    abs(2 * hi - 2 * gh), abs(2 * lo - 2 * gl))
            scale.append(c * h)
    bound = d * math.fsum(scale) + k * gh * d
    bound = bound * (1 + 8 * U) + 0.0
    rel = bound / sk["energy"] if sk["energy"] > 0 else 0.0
    if rel > RATE:
        v = -1
    elif k == 0:
        v = 1
    else:
        v = 0
    return {"gamma": which, "lo": gl, "hi": gh, "ghat": ghat, "k_risk": k,
            "dsse_bound": bound, "rel": rel, "verdict": v}


def evaluate(source, bins: int, memory_max: str = UNIT_MEMORY_MAX,
             share: float = DEFAULT_SHARE, prev: str = GENESIS) -> dict:
    sk = stream_sketch(source, bins)
    rows = []
    key = prev
    for g in GAMMAS:
        c = certify(sk, g)
        text = canon({"gamma": g, "lo": c["lo"].hex(), "hi": c["hi"].hex(),
                      "bins": bins, "k_risk": c["k_risk"], "rel": c["rel"].hex()})
        key = close_rule(key, pack_i2s([c["verdict"]]), text)
        c["key"] = key
        rows.append(c)
    verdicts = [c["verdict"] for c in rows]
    report = {
        "n": sk["n"], "bins": bins, "memory_max": memory_max, "share": share,
        "budget_bytes": budget_bytes(memory_max, share),
        "sketch_bytes": bins * BIN_BYTES, "sort_bytes": sk["n"] * SORT_ELEM_BYTES,
        "sort_fits": sort_fits(sk["n"], memory_max, share),
        "fits": bins * BIN_BYTES <= budget_bytes(memory_max, share),
        "rate": RATE, "verdicts": verdicts,
        "rows": [{kk: (vv.hex() if isinstance(vv, float) else vv) for kk, vv in c.items()}
                 for c in rows],
        "bind": BIND, "ports": list(PORTS), "mint": MINT, "socket": SOCKET,
        "layer_one": LAYER_ONE, "fetch": FETCH, "ue5_launch": UE5_LAUNCH,
    }
    report["key"] = close_rule(key, pack_i2s(verdicts), canon(report))
    return report


def plan_bins(source, memory_max: str = UNIT_MEMORY_MAX, share: float = DEFAULT_SHARE):
    """Smallest power-of-two B within budget with every verdict >= 0, else None."""
    cap = bins_for_budget(memory_max, share)
    b = 8
    while b <= cap:
        rep = evaluate(source, b, memory_max, share)
        if min(rep["verdicts"]) >= 0:
            return b, rep
        b *= 2
    return None, None


# ---- exact sort path (reference, same definitions as WINSOR.md engines) ----
def exact_gamma(w, which: str) -> float:
    a = sorted(abs(x) for x in w)
    n = len(a)
    if which == "absmean":
        return math.fsum(a) / n
    if which == "drop":
        d = n // 16
        t = a[: n - d] if d else a
        return math.fsum(t) / len(t)
    p = a[max(0, math.ceil(0.95 * n) - 1)]
    return math.fsum(min(x, p) for x in a) / n


def quantize(w, gamma: float):
    out = []
    for x in w:
        r = abs(x) / gamma if gamma > 0 else 0.0
        out.append(0 if r < 0.5 else (1 if x > 0 else -1))
    return out


def sse(w, trits, gamma: float) -> float:
    return math.fsum((x - t * gamma) ** 2 for x, t in zip(w, trits))


def synthetic(n: int, seed: int):
    """Irwin-Hall from random.random() only, so it is identical on glibc and musl."""
    rng = random.Random(seed)
    w = []
    for i in range(n):
        x = (rng.random() + rng.random() + rng.random() + rng.random() - 2.0) * 0.02
        if i % 97 == 0:
            x *= 12.0
        w.append(x)
    return w


def selftest() -> int:
    fails = 0

    def ok(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            fails += 1

    ok("flags false, loopback", not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
       and BIND == "127.0.0.1" and len(PORTS) == 5)
    ok("MemoryMax 2G parses base 1024", parse_memory_max("2G") == 2 * 1024 ** 3)
    ok("budget bins at 1/16 of 2G", bins_for_budget() == (2 * 1024 ** 3 // 16) // 40)
    ok("2560x6912 sort does not fit 1/16 of 2G", not sort_fits(2560 * 6912))
    ok("8192-weight sort fits", sort_fits(8192))
    ok("pack_i2s codes", pack_i2s([1, -1, 0, 1]) == bytes([0b01100001])
       and pack_i2s([-1]) == bytes([0b10000000]))

    w = synthetic(8192, 7)
    src = lambda: iter(w)  # noqa: E731
    ex = {g: exact_gamma(w, g) for g in GAMMAS}
    ok("three gammas differ", len({round(v, 12) for v in ex.values()}) == 3)
    energy = math.fsum(x * x for x in w)
    widths = {}
    for bins in (16, 64, 256, 1024, 4096):
        sk = stream_sketch(src, bins)
        for g in GAMMAS:
            c = certify(sk, g)
            inside = c["lo"] <= ex[g] <= c["hi"]
            ok(f"B={bins} {g} bracket holds exact", inside)
            tg = quantize(w, ex[g])
            th = quantize(w, c["ghat"])
            flips = sum(1 for a, b in zip(tg, th) if a != b)
            ok(f"B={bins} {g} flips {flips} <= k_risk {c['k_risk']}", flips <= c["k_risk"])
            dsse = abs(sse(w, tg, ex[g]) - sse(w, th, c["ghat"]))
            ok(f"B={bins} {g} dSSE <= certified", dsse <= c["dsse_bound"] * (1 + 1e-9) + 1e-300)
            ok(f"B={bins} {g} rel uses energy", abs(c["rel"] * energy - c["dsse_bound"])
               <= 1e-9 * max(c["dsse_bound"], 1e-300))
            widths[(bins, g)] = c["hi"] - c["lo"]
    for g in ("drop", "p95"):
        ok(f"{g} bracket tightens 16 -> 4096", widths[(4096, g)] < widths[(16, g)])
    ok("absmean bracket is fp slack only", widths[(16, "absmean")] <= 1e-12 * ex["absmean"])

    b, rep = plan_bins(src)
    ok(f"plan finds B={b} within budget", b is not None and rep["fits"]
       and min(rep["verdicts"]) >= 0)

    r1 = evaluate(src, 256)
    r2 = evaluate(src, 256)
    r3 = evaluate(src, 256, prev="f" * 16)
    ok("key deterministic", r1["key"] == r2["key"] and len(r1["key"]) == 16)
    ok("key depends on prev", r1["key"] != r3["key"])
    ok("report JSON canonical", json.loads(canon(r1))["key"] == r1["key"])

    small = [0.3, -0.1, 0.2, 0.0, -0.4]
    sk = stream_sketch(lambda: iter(small), 4)
    for g in GAMMAS:
        c = certify(sk, g)
        ok(f"n=5 {g} bracket holds", c["lo"] <= exact_gamma(small, g) <= c["hi"])
    ties = [0.01] * 40 + [-0.01] * 40 + [0.5]
    sk = stream_sketch(lambda: iter(ties), 8)
    for g in GAMMAS:
        c = certify(sk, g)
        ok(f"ties {g} bracket holds", c["lo"] <= exact_gamma(ties, g) <= c["hi"])
    zeros = [0.0] * 32
    sk = stream_sketch(lambda: iter(zeros), 8)
    ok("all-zero tensor: gamma 0, verdict defined",
       all(certify(sk, g)["hi"] == 0.0 and certify(sk, g)["verdict"] in (-1, 0, 1)
           for g in GAMMAS))
    try:
        state = {"k": 0}

        def shifty():
            state["k"] += 1
            return iter(w if state["k"] == 1 else w[:-1])
        stream_sketch(shifty, 16)
        ok("source change detected", False)
    except ValueError:
        ok("source change detected", True)

    print(f"selftest: {'PASS' if fails == 0 else 'FAIL'} ({fails} failed)")
    return 0 if fails == 0 else 1


def _file_source(path: str):
    def gen():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield float(line)
    return gen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--weights")
    ap.add_argument("--synthetic", type=int)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bins", type=int)
    ap.add_argument("--memory-max", default=UNIT_MEMORY_MAX)
    ap.add_argument("--share", type=float, default=DEFAULT_SHARE)
    ap.add_argument("--prev", default=GENESIS)
    ap.add_argument("--plan")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.plan:
        rows, cols = (int(v) for v in a.plan.lower().split("x"))
        n = rows * cols
        print(canon({"n": n, "sort_bytes": n * SORT_ELEM_BYTES,
                     "budget_bytes": budget_bytes(a.memory_max, a.share),
                     "sort_fits": sort_fits(n, a.memory_max, a.share),
                     "max_bins": bins_for_budget(a.memory_max, a.share)}))
        return 0
    if a.weights:
        src = _file_source(a.weights)
    elif a.synthetic:
        w = synthetic(a.synthetic, a.seed)
        src = lambda: iter(w)  # noqa: E731
    else:
        ap.print_help()
        return 2
    if a.bins:
        rep = evaluate(src, a.bins, a.memory_max, a.share, a.prev)
    else:
        _, rep = plan_bins(src, a.memory_max, a.share)
        if rep is None:
            print(canon({"error": "no power-of-two B within budget meets RATE"}))
            return 1
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if min(rep["verdicts"]) >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
