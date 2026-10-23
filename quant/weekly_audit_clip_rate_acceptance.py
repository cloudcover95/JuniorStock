#!/usr/bin/env python3
"""Weekly clip-rate acceptance test for the chief weekly audit (JuniorStock engine).

Roadmap anchor: JuniorHome docs/ROADMAP.md, "chief weekly audit" (Sun/Mon 17:00),
remaining engine JuniorStock. This file supplies the quant-error / Winsor-bound
side only; it does not write docs/WEEKLY_AUDIT.md and does not touch
scripts/advance_chief_weekly_write.py (LLC paths).

New capability: a statistical acceptance test. For each port and each Winsor
gamma variant, the week's pooled clip count is tested against the Winsor rate
0.05 with a two-sided Wilson score interval and a one-sided Hoeffding tail
bound, and the week's ternary reconstruction RMS is tested against the
analytic per-element bound. Output is a pass / review verdict per port plus a
close-rule receipt, so the weekly audit can cite a number rather than a trend.

Gamma variants (JuniorHome docs/WINSOR.md, QUANT_ERROR.md):
  absmean     BitNet absmean, gamma = mean |W|
  junior_drop mean |W| after dropping the top 1/16 of |W|
  p95         95th percentile of |W| (clip point)
Ternary quant: q = clamp(round(W / gamma), -1, 1), W_hat = gamma * q.
Per-element error bound: |W - W_hat| <= max(gamma / 2, |W| - gamma).

Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
i2sd userspace reference: 127.0.0.1:8767 (named only; this file opens no socket).

Flags: mint false, socket false, layer_one false, fetch false, ue5_launch false.
Rate 0.05. Close rule: sha256(prev || pack || text)[:16].
Stdlib only. Run with --selftest.
"""
import hashlib
import json
import math
import random
import sys

RATE = 0.05
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
I2SD_ADDR = ("127.0.0.1", 8767)
FLAGS = {"mint": False, "socket": False, "layer_one": False,
         "fetch": False, "ue5_launch": False}
Z95 = 1.959963984540054


def gamma_absmean(w):
    return sum(abs(x) for x in w) / len(w)


def gamma_junior_drop(w):
    a = sorted(abs(x) for x in w)
    keep = len(a) - len(a) // 16
    keep = max(keep, 1)
    return sum(a[:keep]) / keep


def gamma_p95(w):
    a = sorted(abs(x) for x in w)
    k = max(0, min(len(a) - 1, int(math.ceil(0.95 * len(a))) - 1))
    return a[k]


GAMMAS = {"absmean": gamma_absmean, "junior_drop": gamma_junior_drop,
          "p95": gamma_p95}


def ternary_stats(w, gamma):
    """Return (clip_count, sq_err_sum, sq_bound_sum, max_violation)."""
    g = gamma if gamma > 0 else 1e-12
    clips = 0
    se = 0.0
    sb = 0.0
    viol = 0.0
    for x in w:
        q = max(-1, min(1, int(round(x / g))))
        err = abs(x - g * q)
        bound = max(g / 2.0, abs(x) - g)
        if abs(x) > g:
            clips += 1
        se += err * err
        sb += bound * bound
        viol = max(viol, err - bound)
    return clips, se, sb, viol


def wilson(k, n, z=Z95):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def hoeffding_p(k, n, rate=RATE):
    """One-sided P(observed >= k | true rate) bound, exp(-2 n t^2)."""
    if n == 0:
        return 1.0
    t = k / n - rate
    return 1.0 if t <= 0 else math.exp(-2.0 * n * t * t)


def accept_week(daily_batches, rate=RATE):
    """daily_batches: list of weight lists (one per day, up to 7).

    Pools clip counts for p95 (the Winsor-rate variant) and reports ternary
    RMS vs bound RMS for every gamma variant.
    """
    out = {}
    for name, fn in GAMMAS.items():
        k = n = 0
        se = sb = 0.0
        viol = 0.0
        for w in daily_batches:
            c, e, b, v = ternary_stats(w, fn(w))
            k += c
            n += len(w)
            se += e
            sb += b
            viol = max(viol, v)
        lo, hi = wilson(k, n)
        rms = math.sqrt(se / n) if n else 0.0
        brms = math.sqrt(sb / n) if n else 0.0
        row = {"n": n, "clips": k, "clip_frac": round(k / n, 6) if n else 0.0,
               "rms": round(rms, 6), "bound_rms": round(brms, 6),
               "bound_ok": viol <= 1e-9 and rms <= brms + 1e-12}
        if name == "p95":
            row["wilson95"] = [round(lo, 6), round(hi, 6)]
            row["rate_in_ci"] = lo <= rate <= hi
            row["hoeffding_p_over"] = round(hoeffding_p(k, n, rate), 6)
        out[name] = row
    p = out["p95"]
    verdict = "pass" if (all(r["bound_ok"] for r in out.values())
                         and (p["rate_in_ci"] or p["clip_frac"] <= rate)) else "review"
    return {"variants": out, "verdict": verdict}


def close(prev, pack, text):
    return hashlib.sha256((prev + pack + text).encode()).hexdigest()[:16]


def weekly_receipt(per_port_batches, prev="0" * 16):
    ports = {}
    for port in PORTS:
        ports[port] = accept_week(per_port_batches.get(port, []))
    text = json.dumps(ports, sort_keys=True, separators=(",", ":"))
    pack = "juniorstock.weekly_audit.clip_rate_acceptance"
    return {"engine": "JuniorStock", "rate": RATE, "flags": FLAGS,
            "i2sd": "%s:%d" % I2SD_ADDR, "ports": ports,
            "close": close(prev, pack, text), "prev": prev}


def _synthetic(seed, days=7, n=1024, heavy=False):
    rnd = random.Random(seed)
    week = []
    for _ in range(days):
        if heavy:
            w = [rnd.gauss(0, 1) * (8.0 if rnd.random() < 0.2 else 1.0)
                 for _ in range(n)]
        else:
            w = [rnd.gauss(0, 1) for _ in range(n)]
        week.append(w)
    return week


def selftest():
    # Gamma ordering on Gaussian weights: junior_drop <= absmean <= p95.
    w = _synthetic(1, days=1)[0]
    ga, gj, gp = gamma_absmean(w), gamma_junior_drop(w), gamma_p95(w)
    assert gj <= ga <= gp, (gj, ga, gp)
    # Per-element bound always holds.
    for g in (ga, gj, gp):
        assert ternary_stats(w, g)[3] <= 1e-9
    # Wilson interval sanity.
    lo, hi = wilson(50, 1000)
    assert lo < 0.05 < hi
    assert hoeffding_p(30, 1000) == 1.0
    assert hoeffding_p(200, 1000) < 1e-6
    # Clean week passes at rate 0.05 for P95 clipping.
    r = weekly_receipt({p: _synthetic(i + 10) for i, p in enumerate(PORTS)})
    assert set(r["ports"]) == set(PORTS)
    for p in PORTS:
        assert r["ports"][p]["verdict"] == "pass", (p, r["ports"][p])
        assert r["ports"][p]["variants"]["p95"]["rate_in_ci"]
    # Close rule is deterministic, 16 hex, and chains on prev.
    r2 = weekly_receipt({p: _synthetic(i + 10) for i, p in enumerate(PORTS)})
    assert r["close"] == r2["close"] and len(r["close"]) == 16
    r3 = weekly_receipt({p: _synthetic(i + 10) for i, p in enumerate(PORTS)},
                        prev=r["close"])
    assert r3["close"] != r["close"]
    assert I2SD_ADDR[0] == "127.0.0.1"
    assert not any(FLAGS.values())
    print("selftest ok close=%s" % r["close"])
    return 0


def main(argv):
    if "--selftest" in argv:
        return selftest()
    if len(argv) > 1 and argv[1] != "-":
        with open(argv[1]) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)
    prev = data.get("prev", "0" * 16)
    print(json.dumps(weekly_receipt(data.get("ports", {}), prev), indent=2,
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
