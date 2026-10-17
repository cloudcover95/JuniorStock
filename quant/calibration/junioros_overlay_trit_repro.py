"""JuniorOS overlay cross-libc trit reproducibility bound. Stdlib only, offline.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay": Debian/Alpine +
os-release.junior + hardened bitnetd.service + juniorctl on PATH; I2_S 2B4T only
if GGUF already on disk (Quant Chief, pair A, slot 2026-10-17 14:20 MT). This is
the quant-error / Winsor-bound side of that item. The I2_S overlay key side
belongs to Trit Pack & I2_S (JuniorLLM scripts/junioros_overlay_jtr1_key.py).
This file writes no service file and no OS config.

Why: the overlay ships the same bitnetd / i2sd on Debian (glibc) and Alpine
(musl). Ternary trits come from RoundClip(W / gamma, -1, 1), and a weight whose
|W|/gamma sits near the 0.5 threshold can land on a different trit if gamma is
computed with a different summation order or libm on the two bases. Then the
two distros disagree on the trit stream and on every key closed over it.

New capability: a certified cross-distro trit-flip margin. For each of the
three Winsor gammas (JuniorHome docs/WINSOR.md)
  absmean = mean|W|                      (BitNet b1.58)
  drop    = mean|W| after dropping the top 1/16 of |W|   (Junior drop)
  p95     = mean of min(|W|, P95)          (P95 clip, nearest-rank)
and for two summation paths
  fsum    = math.fsum (correctly rounded, so base-independent)
  naive   = left-to-right float adds (what an unpinned build may do)
it bounds the relative gamma error from first principles (Higham:
|s^ - s| <= g_{n-1} s for non-negative terms, g_k = k u / (1 - k u),
u = 2**-53, plus one division rounding), turns that into a ratio margin tau
around the 0.5 threshold, counts at-risk weights k, and bounds the change in
quantization SSE (JuniorHome docs/QUANT_ERROR.md) between the two bases:
  flips:  |dSSE| <= k * 2 tau * gamma^2          (since (g-|w|)^2-|w|^2 = g^2(1-2r))
  scale:  |dSSE| <= sum_{t!=0} (2|e_i| g rho + g^2 rho^2)   (rho = rel gamma error)
Sorting and the P95 / 1/16 cut indices are exact, so only summation drifts.
Verdict trit per (gamma, path): +1 no weight at risk (trit stream provably
identical on Debian and Alpine), 0 at risk but relative dSSE <= RATE (0.05),
-1 relative dSSE > RATE. Empirically it also measures forward-vs-reverse naive
sums and asserts observed flips never exceed the certified count.

Synthetic weights use only random.random() (Mersenne Twister, exact on every
base) via an Irwin-Hall sum, never gauss/log, so the selftest itself is
libc-independent.

Close rule: sha256(prev || pack || text)[:16]. Per gamma: pack = I2_S 2-bit
trit codes of the fsum trits (0->0, 1->1, -1->2, MSB first), text = gamma.hex().
Report key: pack = I2_S code of the 6 verdict trits, text = canonical JSON.
Rate 0.05 is the SSE drift budget. GENESIS prev = "0" * 16.
Compatible engines: BitNet-mlx, JuniorLLM, JuniorQuant, bitnetd 127.0.0.1:8765,
i2sd 127.0.0.1:8767 (named, never dialed), JuniorOS overlay (Debian/Alpine).
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no team-module import.

Usage:
  python3 junioros_overlay_trit_repro.py --selftest
  python3 junioros_overlay_trit_repro.py --weights FILE [--prev HEX16]
  python3 junioros_overlay_trit_repro.py --synthetic 4096 --seed 7
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
PATHS = ("fsum", "naive")
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


def g_k(k: int) -> float:
    ku = k * U
    if ku >= 1.0:
        return math.inf
    return ku / (1.0 - ku)


def _naive_sum(xs) -> float:
    s = 0.0
    for x in xs:
        s += x
    return s


def winsor_terms(w, which: str):
    """Non-negative terms whose mean is the requested gamma (exact selection)."""
    a = sorted(abs(x) for x in w)
    n = len(a)
    if which == "absmean":
        return [abs(x) for x in w]
    if which == "drop":
        d = n // 16
        return a[: n - d] if d else a
    if which == "p95":
        p = a[max(0, math.ceil(0.95 * n) - 1)]
        return [min(abs(x), p) for x in w]
    raise ValueError(which)


def gamma_of(terms, path: str, reverse: bool = False) -> float:
    xs = list(reversed(terms)) if reverse else terms
    s = math.fsum(xs) if path == "fsum" else _naive_sum(xs)
    return s / len(xs)


def rel_bound(n_terms: int, path: str) -> float:
    """Certified |gamma^/gamma - 1| for non-negative terms."""
    if path == "fsum":
        return (1.0 + U) ** 2 - 1.0
    return (1.0 + g_k(max(n_terms - 1, 0))) * (1.0 + U) - 1.0


def quantize(w, gamma: float):
    out = []
    for x in w:
        r = abs(x) / gamma if gamma > 0 else 0.0
        out.append(0 if r < 0.5 else (1 if x > 0 else -1))
    return out


def sse(w, trits, gamma: float) -> float:
    return math.fsum((x - t * gamma) ** 2 for x, t in zip(w, trits))


def analyse(w, which: str, path: str) -> dict:
    terms = winsor_terms(w, which)
    gam = gamma_of(terms, path)
    rho = rel_bound(len(terms), path)
    tau = 0.5 * (rho + 2 * U) / (1.0 - rho) + U
    trits = quantize(w, gam)
    at_risk = sum(1 for x in w if abs(abs(x) / gam - 0.5) <= tau) if gam > 0 else 0
    base = sse(w, trits, gam)
    d_flip = at_risk * 2.0 * tau * gam * gam
    d_scale = math.fsum(
        2.0 * abs(x - t * gam) * gam * rho + gam * gam * rho * rho
        for x, t in zip(w, trits) if t
    )
    d_bound = d_flip + d_scale
    rel_d = d_bound / base if base > 0 else (0.0 if d_bound == 0 else math.inf)
    if at_risk == 0:
        verdict = 1
    elif rel_d <= RATE:
        verdict = 0
    else:
        verdict = -1
    g_rev = gamma_of(terms, path, reverse=True)
    t_rev = quantize(w, g_rev)
    flips_obs = sum(1 for a, b in zip(trits, t_rev) if a != b)
    d_obs = abs(sse(w, t_rev, g_rev) - base)
    return {
        "gamma": which, "path": path, "n_terms": len(terms),
        "gamma_hex": gam.hex(), "gamma_rev_hex": g_rev.hex(),
        "rho": rho, "tau": tau, "at_risk": at_risk,
        "flips_observed": flips_obs, "sse": base,
        "dsse_bound": d_bound, "dsse_observed": d_obs, "rel_dsse_bound": rel_d,
        "verdict": verdict,
        "trit_key": close_rule(GENESIS, pack_i2s(trits), gam.hex()),
        "_trits": trits,
    }


def evaluate(w, prev: str = GENESIS) -> dict:
    if len(w) == 0:
        raise ValueError("no weights")
    rows = [analyse(w, g, p) for g in GAMMAS for p in PATHS]
    for r in rows:
        r.pop("_trits")
    verdicts = [r["verdict"] for r in rows]
    body = {
        "item": "BETA_TO_OS.md JuniorOS overlay (quant-error side)",
        "n": len(w), "rate": RATE, "bind": BIND,
        "engines": ["BitNet-mlx", "JuniorLLM", "JuniorQuant",
                    "bitnetd %s:%d" % (BIND, BITNETD_PORT),
                    "i2sd %s:%d" % (BIND, I2SD_PORT), "JuniorOS overlay"],
        "ports": list(PORTS), "rows": rows, "verdicts": verdicts,
        "recommend": "pin gamma to math.fsum on Debian and Alpine"
        if all(r["verdict"] == 1 for r in rows if r["path"] == "fsum") else
        "weights sit on the 0.5 threshold; pin a tie rule before shipping",
        "flags": {"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
                  "fetch": FETCH, "ue5_launch": UE5_LAUNCH},
    }
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    body["prev"] = prev
    body["close"] = close_rule(prev, pack_i2s(verdicts), text)
    return body


def synthetic(n: int, seed: int):
    rng = random.Random(seed)
    w = []
    for i in range(n):
        x = (rng.random() + rng.random() + rng.random() + rng.random() - 2.0) * 0.02
        if i % 97 == 0:
            x *= 12.0  # heavy-tail outliers so the three gammas differ
        w.append(x)
    return w


def selftest() -> int:
    fails = 0

    def ok(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            fails += 1

    ok("flags false", not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH))
    ok("bind loopback", BIND == "127.0.0.1")
    ok("rate 0.05", RATE == 0.05)
    ok("ports named", PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"))
    ok("pack codes", pack_i2s([0, 1, -1, 0]) == bytes([0b00011000])
       and pack_i2s([1]) == b"\x40")
    exp = hashlib.sha256(GENESIS.encode() + b"\x40" + b"t").hexdigest()[:16]
    ok("close rule matches team form", close_rule(GENESIS, b"\x40", "t") == exp)

    v = [float(i) for i in range(1, 33)]  # |W| = 1..32
    ok("absmean", gamma_of(winsor_terms(v, "absmean"), "fsum") == 16.5)
    ok("drop top 1/16", gamma_of(winsor_terms(v, "drop"), "fsum") == 15.5)
    p95 = 31.0  # nearest rank ceil(0.95*32)=31
    ok("p95 clip", gamma_of(winsor_terms(v, "p95"), "fsum")
       == math.fsum(min(x, p95) for x in v) / 32)
    ok("fsum rel bound ~2u", rel_bound(10 ** 6, "fsum") < 3 * U)
    ok("naive rel bound grows with n", rel_bound(10 ** 6, "naive") > 1e5 * U)

    # certified bounds hold on random sets (observed <= bound)
    held = True
    for seed in range(6):
        w = synthetic(3000 + 517 * seed, seed)
        for g in GAMMAS:
            for p in PATHS:
                r = analyse(w, g, p)
                g0 = math.fsum(winsor_terms(w, g)) / r["n_terms"]
                if abs(float.fromhex(r["gamma_hex"]) / g0 - 1) > r["rho"]:
                    held = False
                if r["flips_observed"] > r["at_risk"]:
                    held = False
                if r["dsse_observed"] > r["dsse_bound"] * (1 + 1e-9) + 1e-300:
                    held = False
    ok("gamma, flip and dSSE bounds hold", held)

    # adversarial: weights placed on the threshold of the naive gamma
    w = synthetic(2048, 11)
    c = 0.5 * gamma_of(winsor_terms(w, "absmean"), "naive")
    for _ in range(8):  # fixed point: extra weights sit at 0.5 * gamma(w2)
        w2 = w + [c * (1 + k * 1e-15) for k in range(-40, 41)]
        c = 0.5 * gamma_of(winsor_terms(w2, "absmean"), "naive")
    rn = analyse(w2, "absmean", "naive")
    ok("threshold weights flagged at risk", rn["at_risk"] > 0 and rn["verdict"] <= 0)
    ok("fsum margin far tighter than naive",
       analyse(w2, "absmean", "fsum")["tau"] < rn["tau"] / 100)

    w = synthetic(4096, 7)
    a = evaluate(w)
    ok("six verdict trits", len(a["verdicts"]) == 6
       and all(t in (-1, 0, 1) for t in a["verdicts"]))
    ok("three gammas differ", len({r["gamma_hex"] for r in a["rows"]
                                   if r["path"] == "fsum"}) == 3)
    ok("close deterministic", evaluate(w)["close"] == a["close"] and len(a["close"]) == 16)
    ok("close prev-sensitive", evaluate(w, prev="f" * 16)["close"] != a["close"])
    ok("trit key changes with stream", evaluate(w[:-1])["rows"][0]["trit_key"]
       != a["rows"][0]["trit_key"])
    ok("no socket in module", "socket" not in globals())
    print("selftest %s (%d fail)" % ("ok" if fails == 0 else "FAILED", fails))
    return 0 if fails == 0 else 1


def _read_weights(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read().replace(",", " ").split()
    return [float(x) for x in raw]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--weights")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.weights:
        w = _read_weights(a.weights)
    elif a.synthetic > 0:
        w = synthetic(a.synthetic, a.seed)
    else:
        ap.error("need --selftest, --weights or --synthetic")
    print(json.dumps(evaluate(w, a.prev), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
