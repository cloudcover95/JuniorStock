"""ROADMAP automations "19:00 second slice if receipt >45 min old": certified trit gate.

Stdlib only, offline, read-only. Ports touched: Home, OSai, FieldCore,
Flagstaff, JuniorLLM. TCP ports named, never dialed: bitnetd 127.0.0.1:8765,
i2sd userspace 127.0.0.1:8767. Bind 127.0.0.1 only; this file opens no socket.
mint false, socket false, layer_one false, fetch false, ue5_launch false.

Roadmap item: JuniorHome docs/ROADMAP.md, "Automations (current)", the line
"19:00 - second slice if receipt >45 min old" (with docs/AUDIT_SLOTS.md
"skip_overlap if LAST_RECEIPT <45 min"). Quant Chief, pair A, 2026-10-19
14:20 MT: JuniorOS overlay remaining gaps only, else Verifier's fallback, the
one roadmap item with no quant-team file: this one. Quant-error / Winsor side.
This file is not an LLC path (it is not web3node/ and writes no JuniorHome
doc); it never schedules, runs or skips a slice itself. It only certifies the
decision the 19:00 slot has to make.

The gap: the 19:00 gate compares "now - LAST_RECEIPT" with 45 min, but the
receipts are quantized. LAST_RECEIPT is written at minute resolution
("2026-10-11T19:14-06:00", truncated), the audit table stamps are minutes, and
the slot start itself jitters (WEEKLY_AUDIT 19:00 rows land at 19:11..19:28).
Nothing bounds that quantization error, so near the threshold the gate can
run a second slice on a receipt that is really 44 min old (an overlap), or
skip one that is really 46 min old.

New capability:
  1. Certified age interval. A stamp with resolution q (60 s without seconds,
     1 s with seconds, 10^-k s with k fraction digits) is a floor, so the true
     time is in [t, t + q). Hence the true age is in the OPEN interval
       (obs - q_receipt - skew, obs + q_now + skew)
     for a declared clock-skew bound skew >= 0 (default 0). All in integer
     microseconds, exact.
  2. Gate trit g against T = 45 min ("> T" is the rule):
       +1 second_slice   lo >= T  (age > lo >= T, certainly older)
       -1 skip_overlap   hi <= T  (age < hi <= T, certainly not older)
        0 skip_overlap_fail_closed (the interval straddles T; the
          quantization error alone could flip the rule, so do not build)
     resolution_ok = (hi - lo) / T <= RATE 0.05 (minute stamps: 121 s / 2700 s).
  3. Winsor jitter margin. From the slot's start lags l_i (receipt minute -
     slot minute, WEEKLY_AUDIT), deviations d_i = l_i - lower median. For each
     Winsor gamma (JuniorHome docs/WINSOR.md, engine-matched to JuniorLLM
     junior_bitnet):
       absmean  gamma = mean|d|                     (junior_bitnet.math.absmean)
       drop     gamma = mean|d| after dropping the top n // 16 (== absmean, n < 16)
       clip     tau = sorted|d|[floor(0.95 (n - 1))], gamma = mean min(|d|, tau),
                trit uses gamma + 1e-7               (junior_bitnet.winsor.pack)
     The lags are minute floors too, so each true d_i is within (-1, 1) min of
     the observed one; every gamma above is 1-Lipschitz in the L-inf norm
     (order statistics, min and mean all are), so the true gamma is in
     (gamma - 1, gamma + 1) min. With the RATE slack, gamma_hi = (gamma + 1)(1 + 0.05).
     BitNet round-clip zeroes |x| <= gamma / 2, so the variant trit is
       +1 if lo - T >= gamma_hi / 2, -1 if hi - T <= -gamma_hi / 2, else 0
     (the certified margin clears half a worst-case jitter gamma), plus a
     RATE stability flag (trit of the observed margin unchanged at gamma (1 -+ 0.05)).
  4. Quant error of the jitter series under each gamma: SSE = sum (d - gamma t)^2,
     rel = SSE / sum d^2, and the zero-band share (JuniorHome docs/QUANT_ERROR.md).
  robust = g != 0 and every variant trit == g.

Rows: one per variant, then the gate row. Row trits: variant [v, stable, g],
gate [g, robust, resolution_ok]. I2_S pack: codes 0->0, 1->1, -1->2, MSB
first, zero padded. Close rule: key = sha256(prev || pack || text)[:16],
GENESIS prev = "0" * 16 as ASCII hex, text = canonical JSON of the row.
Rate lock: sha256(prev || pack || b"0.05").hexdigest() beside the final key.

It does not wrap, test, or re-derive any team file or earlier Winsor file
(JuniorQuant src/winsor_clip.py, src/i2sd_*, src/live_beta_*; JuniorStock
risk/winsor_quant_error.py, security/*, quant/calibration/*.py).

Run: python3 slot1900_receipt_age_trit_gate.py --selftest
     python3 slot1900_receipt_age_trit_gate.py --now 2026-10-19T19:00:30-06:00 \
         --receipt 2026-10-19T18:15-06:00 [--lags 15,12,12,11,28,14,12,14] [--skew-s 0]
Compatible engines: JuniorQuant, JuniorStock, JuniorLLM junior_bitnet
(absmean, winsor.pack), BitNet-mlx, bitnetd, i2sd (same I2_S code map and close rule).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime
from math import floor

RATE = 0.05
GENESIS = "0" * 16
THRESHOLD_MIN = 45
ENGINE_EPS = 1e-7
VARIANTS = ("absmean", "drop", "clip")
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
TCP = {"bitnetd": "127.0.0.1:8765", "i2sd": "127.0.0.1:8767"}
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
CODE = {0: 0, 1: 1, -1: 2}
US = 1_000_000
# JuniorHome docs/WEEKLY_AUDIT.md, 19:00 rows 2026-10-05 .. 2026-10-12 (minutes after 19:00).
SEED_LAGS = (15, 12, 12, 11, 28, 14, 12, 14)


# ---------------------------------------------------------------- pack / close
def pack(trits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(trits), 4):
        chunk = list(trits[i:i + 4]) + [0] * (4 - len(trits[i:i + 4]))
        b = 0
        for t in chunk:
            b = (b << 2) | CODE[t]
        out.append(b)
    return bytes(out)


def close_rule(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def rate_lock(prev: str, pk: bytes) -> str:
    return hashlib.sha256(prev.encode() + pk + str(RATE).encode()).hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------- stamps
_TIME = re.compile(r"T\d{2}:\d{2}(?::(\d{2})(?:\.(\d+))?)?")


def stamp(s: str) -> tuple[int, int]:
    """(epoch microseconds of the floor, resolution q in microseconds)."""
    m = _TIME.search(s)
    if not m:
        raise ValueError(f"no time in stamp {s!r}")
    if m.group(1) is None:
        q = 60 * US
    elif m.group(2) is None:
        q = US
    else:
        q = max(1, US // (10 ** min(6, len(m.group(2)))))
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        raise ValueError(f"stamp {s!r} has no UTC offset")
    whole = int(dt.timestamp())
    return whole * US + dt.microsecond, q


def age_interval(now: str, receipt: str, skew_s: float = 0.0) -> dict:
    n, qn = stamp(now)
    r, qr = stamp(receipt)
    skew = int(round(skew_s * US))
    if skew < 0:
        raise ValueError("skew must be >= 0")
    obs = n - r
    return {"obs_us": obs, "lo_us": obs - qr - skew, "hi_us": obs + qn + skew,
            "q_now_us": qn, "q_receipt_us": qr, "skew_us": skew}


def gate_trit(lo: int, hi: int, t_us: int) -> int:
    if lo >= t_us:
        return 1
    if hi <= t_us:
        return -1
    return 0


ACTION = {1: "second_slice", -1: "skip_overlap", 0: "skip_overlap_fail_closed"}


# ---------------------------------------------------------------- winsor gammas
def clip_round(x: float) -> int:
    return max(-1, min(1, int(round(x))))


def gammas(d: list[float]) -> dict:
    a = [abs(x) for x in d]
    n = len(a)
    if n == 0:
        return {"absmean": 1.0, "drop": 1.0, "clip": 1.0, "tau": 0.0}
    absmean = sum(a) / n or 1.0
    k = n // 16
    kept = sorted(a)[: n - k]
    drop = sum(kept) / len(kept) or 1.0
    s = sorted(a)
    tau = s[min(n - 1, max(0, floor(0.95 * (n - 1))))]
    clip = sum(min(x, tau) for x in a) / n
    return {"absmean": absmean, "drop": drop, "clip": clip, "tau": tau}


def eff(variant: str, g: float) -> float:
    return g + ENGINE_EPS if variant == "clip" else g


def quant_error(d: list[float], g: float) -> dict:
    t = [clip_round(x / g) for x in d]
    sse = sum((x - g * ti) ** 2 for x, ti in zip(d, t))
    en = sum(x * x for x in d)
    return {"sse": round(sse, 9), "rel": round(sse / en, 9) if en else 0.0,
            "zero_share": round(t.count(0) / len(t), 9) if t else 0.0}


def lower_median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[(len(s) - 1) // 2]


def variant_trit(m_lo: float, m_hi: float, g_hi: float) -> int:
    if m_lo > 0 and m_lo >= g_hi / 2:
        return 1
    if m_hi < 0 and -m_hi >= g_hi / 2:
        return -1
    return 0


# ---------------------------------------------------------------- report
def report(now: str, receipt: str, lags=SEED_LAGS, skew_s: float = 0.0,
           threshold_min: int = THRESHOLD_MIN, prev: str = GENESIS) -> dict:
    iv = age_interval(now, receipt, skew_s)
    t_us = threshold_min * 60 * US
    g = gate_trit(iv["lo_us"], iv["hi_us"], t_us)
    res_ok = (iv["hi_us"] - iv["lo_us"]) <= RATE * t_us
    lags = [float(x) for x in lags]
    med = lower_median(lags) if lags else 0.0
    d = [x - med for x in lags]
    gm = gammas(d)
    m_lo = (iv["lo_us"] - t_us) / (60 * US)
    m_hi = (iv["hi_us"] - t_us) / (60 * US)
    m_obs = (iv["obs_us"] - t_us) / (60 * US)
    rows, trits_all = [], []
    vts = []
    for v in VARIANTS:
        ge = eff(v, gm[v])
        g_hi = (ge + 1.0) * (1 + RATE)
        vt = variant_trit(m_lo, m_hi, g_hi)
        base = clip_round(m_obs / ge)
        stable = all(clip_round(m_obs / (ge * f)) == base for f in (1 - RATE, 1 + RATE))
        vts.append(vt)
        row = {"kind": "variant", "variant": v, "gamma_min": round(gm[v], 9),
               "gamma_hi_min": round(g_hi, 9), "margin_lo_min": round(m_lo, 6),
               "margin_hi_min": round(m_hi, 6), "trit": vt, "stable": stable,
               "qe": quant_error(d, ge)}
        tr = [vt, 1 if stable else -1, g]
        pk = pack(tr)
        row["prev"] = prev
        key = close_rule(prev, pk, canon(row))
        row["key"] = key
        rows.append(row)
        trits_all.append(tr)
        prev = key
    robust = g != 0 and all(v == g for v in vts)
    grow = {"kind": "gate", "rule": f"second slice if receipt >{threshold_min} min old",
            "now": now, "receipt": receipt, "interval_us": [iv["lo_us"], iv["hi_us"]],
            "obs_us": iv["obs_us"], "q_us": [iv["q_receipt_us"], iv["q_now_us"]],
            "skew_us": iv["skew_us"], "trit": g, "action": ACTION[g], "robust": robust,
            "resolution_ok": res_ok, "lags_min": lags, "median_min": med,
            "tau_min": gm["tau"], "rate": RATE}
    tr = [g, 1 if robust else (0 if g == 0 else -1), 1 if res_ok else -1]
    pk = pack(tr)
    grow["prev"] = prev
    key = close_rule(prev, pk, canon(grow))
    grow["key"] = key
    rows.append(grow)
    return {"rows": rows, "key": key, "rate_lock": rate_lock(key, pk),
            "action": ACTION[g], "trit": g, "robust": robust, "ports": list(PORTS),
            "tcp": TCP, "flags": FLAGS}


# ---------------------------------------------------------------- selftest
def selftest() -> str:
    assert pack([1, -1, 0, 1]) == bytes([0b01100001])
    assert pack([-1]) == bytes([0b10000000])
    assert close_rule(GENESIS, b"\x02", "x") == hashlib.sha256(b"0" * 16 + b"\x02x").hexdigest()[:16]

    # stamp resolution
    assert stamp("2026-10-11T19:14-06:00")[1] == 60 * US
    assert stamp("2026-10-11T19:14:05-06:00")[1] == US
    assert stamp("2026-10-11T19:14:05.25-06:00")[1] == 10_000
    assert stamp("2026-10-11T19:14-06:00")[0] == stamp("2026-10-12T01:14+00:00")[0]

    # WEEKLY_AUDIT 2026-10-09 19:28: "Receipt 11 min old, skip overlap"
    r = report("2026-10-09T19:28-06:00", "2026-10-09T19:17-06:00")
    assert r["trit"] == -1 and r["action"] == "skip_overlap" and r["robust"]
    assert r["rows"][-1]["interval_us"] == [10 * 60 * US, 12 * 60 * US]

    # AUTOMATION_AUDIT 2026-10-12 16:45: LAST_RECEIPT 2026-10-11T19:14 older than 45 min
    r2 = report("2026-10-12T16:45-06:00", "2026-10-11T19:14-06:00")
    assert r2["trit"] == 1 and r2["action"] == "second_slice" and r2["robust"]

    # straddle: observed 45.5 min, quantization error alone could flip it
    r3 = report("2026-10-19T19:00:30-06:00", "2026-10-19T18:15-06:00")
    assert r3["trit"] == 0 and r3["action"] == "skip_overlap_fail_closed" and not r3["robust"]
    # edge: lo == T exactly is still certainly > T (open interval)
    r4 = report("2026-10-19T19:00:00-06:00", "2026-10-19T18:14-06:00")
    assert r4["trit"] == 1
    assert report("2026-10-19T19:00:00-06:00", "2026-10-19T18:14-06:00", skew_s=1)["trit"] == 0
    # hi == T exactly is certainly not > T
    assert report("2026-10-19T18:59:59-06:00", "2026-10-19T18:15:00-06:00")["trit"] == -1
    # certified but not robust: 46.5 min, margin 1.5 min < gamma_hi / 2
    r5 = report("2026-10-19T19:01:30-06:00", "2026-10-19T18:15:00-06:00")
    assert r5["trit"] == 1 and not r5["robust"]
    # resolution: minute stamps 121 s / 2700 s <= 0.05; 2.5 min skew is not
    assert r["rows"][-1]["resolution_ok"]
    assert not report("2026-10-09T19:28-06:00", "2026-10-09T19:17-06:00", skew_s=75)["rows"][-1]["resolution_ok"]

    # Winsor gammas on the WEEKLY_AUDIT lags: d = [3,0,0,-1,16,2,0,2]
    d = [x - lower_median(list(SEED_LAGS)) for x in SEED_LAGS]
    assert d == [3, 0, 0, -1, 16, 2, 0, 2]
    gm = gammas(d)
    assert gm["absmean"] == 3.0 and gm["drop"] == 3.0 and gm["tau"] == 3 and gm["clip"] == 1.375
    # engine match: clip trits == junior_bitnet.winsor.pack rule, absmean == math.absmean rule
    assert [clip_round(x / (gm["clip"] + ENGINE_EPS)) for x in d] == [1, 0, 0, -1, 1, 1, 0, 1]
    assert [clip_round(x / gm["absmean"]) for x in d] == [1, 0, 0, 0, 1, 1, 0, 1]
    # drop differs from absmean once n >= 16
    big = list(range(16))
    gb = gammas(big)
    assert gb["drop"] == sum(range(15)) / 15 and gb["drop"] < gb["absmean"]

    # 1-Lipschitz: minute-floor errors in [0, 1) move every gamma by < 1 min
    rng = random.Random(1420)
    for _ in range(300):
        e = [rng.random() for _ in SEED_LAGS]
        lt = [a + b for a, b in zip(SEED_LAGS, e)]
        dt = [x - sorted(lt)[(len(lt) - 1) // 2] for x in lt]
        gt = gammas(dt)
        for v in VARIANTS:
            assert abs(gt[v] - gm[v]) < 1.0  # lag and median floors both in [0, 1): |dd_i| < 1
            assert abs(gt[v] - gm[v]) <= max(abs(a - b) for a, b in zip(dt, d)) + 1e-12

    # quant error sanity
    qe = quant_error(d, gm["absmean"])
    assert qe["sse"] == 172.0 and abs(qe["rel"] - 172 / 274) < 1e-9

    # chain: deterministic, tamper-evident, prev threads through
    again = report("2026-10-09T19:28-06:00", "2026-10-09T19:17-06:00")
    assert again["key"] == r["key"]
    assert report("2026-10-09T19:28-06:00", "2026-10-09T19:17-06:00", lags=(15, 12))["key"] != r["key"]
    for a, b in zip(r["rows"], r["rows"][1:]):
        assert b["prev"] == a["key"]
    assert len(r["key"]) == 16 and len(r["rate_lock"]) == 64

    # flags and loopback
    assert not any(FLAGS.values())
    assert all(v.startswith("127.0.0.1:") for v in TCP.values())
    k = hashlib.sha256((r["key"] + r2["key"] + r3["key"]).encode()).hexdigest()[:16]
    return k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--now")
    ap.add_argument("--receipt")
    ap.add_argument("--lags", default=",".join(str(x) for x in SEED_LAGS))
    ap.add_argument("--skew-s", type=float, default=0.0)
    ap.add_argument("--threshold-min", type=int, default=THRESHOLD_MIN)
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args()
    if a.selftest:
        print(json.dumps({"selftest": "pass", "key": selftest(), "rate": RATE}))
        return
    if not (a.now and a.receipt):
        ap.error("--now and --receipt are required (or --selftest)")
    lags = [float(x) for x in a.lags.split(",") if x.strip()]
    print(json.dumps(report(a.now, a.receipt, lags, a.skew_s, a.threshold_min, a.prev), indent=2))


if __name__ == "__main__":
    main()
