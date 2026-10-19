"""Decimal-digits certificate for a Winsor gamma published in os-release.junior.

Stdlib only, offline, read-only. Ports touched: Home, OSai, FieldCore,
Flagstaff, JuniorLLM. TCP ports named, never dialed: bitnetd 127.0.0.1:8765,
i2sd 127.0.0.1:8767.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay", the
`os-release.junior` part (JuniorLLM rails/linux/os-release.junior, copied by
install-overlay.sh to /etc and /usr/lib). Quant Chief, pair A, slot
2026-10-18 18:20 MT, "os-release.junior gaps only". This is the quant-error /
Winsor-bound side. It never writes an OS config file: it reads os-release text,
certifies Winsor/quant profile fields, and prints proposed KEY=VALUE lines.
The os-release file itself is covered by stocksnode f49707c; PATH, Debian,
MemoryMax and user-unit parity already landed; LLC owns advance_junioros_overlay.

The gap: os-release.junior is plain KEY="VALUE" text, so a Winsor gamma
published there (JUNIOR_WINSOR_GAMMA) is a decimal string. Every engine parses
it back: Python and JuniorQuant to f64, bitnetd / BitNet-mlx / i2sd style
readers to f32 (strtof). The ternary trit of a weight is sign(w) iff
2|w| > gamma, else 0, so a gamma that comes back a little off moves the trit
boundary and can flip weights that sit near gamma/2. Nothing so far says how
many significant digits the field needs, or whether an f32 reader can ever
agree with the producer.

New capability, per Winsor variant (absmean = mean|W|; drop = mean |W| after
dropping the top 1/16 by magnitude; clip = mean min(|W|, P95), nearest rank):
  exact    gamma is the exact rational mean of the f32 weights (Fraction).
  publish  dec_d(gamma) = gamma rounded to d significant digits (half-even),
           d = 1..17, exact Fraction; that string is the field value.
  readers  f64: nearest double of the decimal; f32: nearest float32 of the
           decimal, both computed exactly (correct rounding, no double round).
  flips    count of w with trit(w, reader gamma) != trit(w, exact gamma).
  dSSE     |SSE(reader) - SSE(exact)| / SSE(exact), SSE = sum (w - g*t)^2.
  margin   m = min_w | |w| - gamma/2 |; a tensor-free certificate: zero flips
           whenever |g_reader - gamma| < 2m, and |dec_d - gamma| <= 5*10^(e-d)
           with e = floor(log10 gamma) (f32 adds at most half an f32 ulp).
  d_min    smallest d such that d and every longer field (d..17) give zero
           flips and dSSE <= RATE, per reader. Stability matters: a short
           field can dodge a near-boundary weight by luck and a longer one
           flip it, so the first zero-flip d alone is not a certificate.
Verdict trit per variant: +1 both readers reach zero flips at d <= 9 (nine
digits round-trip any float32); 0 the f64 reader does at some d <= 17 but the
f32 reader never does (gamma itself is not safe in f32 for this tensor);
-1 neither, or dSSE > RATE at the chosen d. If os-release text already holds
JUNIOR_WINSOR_VARIANT and JUNIOR_WINSOR_GAMMA, that published value is
certified too (stale-field flips under both readers). JUNIOR_BITNETD must be
loopback and JUNIOR_QUANT_RATE, when present, must equal 0.05 exactly as a
decimal, or the report is red.

Close rule: sha256(prev || pack || text)[:16], GENESIS prev = "0" * 16. Per
variant row: pack = I2_S 2-bit code of its verdict trit (0->0, 1->1, -1->2,
MSB first), text = canonical JSON of the row. Rows chain in VARIANTS order,
then the report key: pack = I2_S code of all row verdicts plus the field trit,
text = canonical JSON of the report. Rate 0.05 is the dSSE budget.
Compatible engines: Python f64 readers (JuniorQuant, JuniorStock, JuniorLLM),
f32 readers (bitnetd 127.0.0.1:8765, BitNet-mlx, i2sd 127.0.0.1:8767), JuniorOS
overlay (Debian/Alpine). Bind 127.0.0.1 only. mint/socket/layer_one/fetch/
ue5_launch false. No model pull, no GGUF download, no network, no subprocess,
no team import, no file writes.

Usage:
  python3 os_release_winsor_digits_cert.py --selftest
  python3 os_release_winsor_digits_cert.py [--os-release PATH] \
      [--weights RAW_F32_LE_FILE] [--seed 7] [--n 4096] [--prev HEX16]
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
RATE_TEXT = "0.05"
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
VARIANTS = ("absmean", "drop", "clip")
F32_DIGITS = 9
MAX_DIGITS = 17
_ENC = {0: 0, 1: 1, -1: 2}

# Snapshot of JuniorLLM rails/linux/os-release.junior (read-only reference).
DEFAULT_OS_RELEASE = '''NAME="JuniorOS"
ID=junioros
ID_LIKE="debian alpine"
PRETTY_NAME="JuniorOS overlay (BitNet rail)"
VARIANT="overlay"
VARIANT_ID=overlay
HOME_URL="https://github.com/cloudcover95/JuniorLLM"
SUPPORT_URL="https://github.com/cloudcover95/JuniorLLM"
JUNIOR_BITNETD="127.0.0.1:8765"
JUNIOR_OFFLINE="1"
JUNIOR_CHARTER="docs/BETA_TO_OS.md"
# Overlay only. Not a custom kernel. No OSWorld score.
'''


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


# ---- os-release parsing (os-release(5) quoting, read-only) -----------------
def parse_os_release(text: str) -> dict:
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if not k.replace("_", "").isalnum():
            continue
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            q, v = v[0], v[1:-1]
            if q == '"':
                res, i = [], 0
                while i < len(v):
                    if v[i] == "\\" and i + 1 < len(v) and v[i + 1] in '"\\`$':
                        res.append(v[i + 1])
                        i += 2
                    else:
                        res.append(v[i])
                        i += 1
                v = "".join(res)
        out[k] = v
    return out


def loopback_ok(hostport: str) -> bool:
    host = hostport.rsplit(":", 1)[0] if ":" in hostport else hostport
    return host == BIND


# ---- exact decimal and float rounding --------------------------------------
def parse_decimal(s: str) -> Fraction | None:
    s = s.strip()
    try:
        if not s or s.lower() in ("nan", "inf", "-inf", "+inf", "infinity"):
            return None
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _round_half_even(x: Fraction) -> int:
    f = math.floor(x)
    r = x - f
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and f % 2 == 1):
        return f + 1
    return f


def _exp10(x: Fraction) -> int:
    """floor(log10 x) for x > 0, exact."""
    e = len(str(x.numerator)) - len(str(x.denominator))
    while Fraction(10) ** e > x:
        e -= 1
    while Fraction(10) ** (e + 1) <= x:
        e += 1
    return e


def round_sig(x: Fraction, d: int) -> tuple[Fraction, str]:
    """x > 0 rounded to d significant digits (half-even), plus its text."""
    e = _exp10(x)
    scale = Fraction(10) ** (e - d + 1)
    m = _round_half_even(x / scale)
    if m >= 10 ** d:
        m //= 10
        e += 1
    k = e - d + 1
    text = f"{m}e{k}" if k else str(m)
    return Fraction(m) * Fraction(10) ** k, text


def nearest_binary(x: Fraction, mant_bits: int, emin: int) -> Fraction:
    """Correctly rounded (half-even) binary float of x > 0, no overflow check."""
    e = x.numerator.bit_length() - x.denominator.bit_length()
    while Fraction(2) ** e > x:
        e -= 1
    while Fraction(2) ** (e + 1) <= x:
        e += 1
    e = max(e, emin)
    ulp = Fraction(2) ** (e - mant_bits + 1)
    return _round_half_even(x / ulp) * ulp


def reader_value(x: Fraction, reader: str) -> Fraction:
    if reader == "f64":
        return nearest_binary(x, 53, -1022)
    return nearest_binary(x, 24, -126)


def f32(v: float) -> float:
    return struct.unpack("<f", struct.pack("<f", v))[0]


def half_ulp32(x: Fraction) -> Fraction:
    e = x.numerator.bit_length() - x.denominator.bit_length()
    while Fraction(2) ** e > x:
        e -= 1
    while Fraction(2) ** (e + 1) <= x:
        e += 1
    return Fraction(2) ** (max(e, -126) - 24)


# ---- Winsor gammas (exact) and ternary error --------------------------------
def gamma_exact(ws, variant: str) -> Fraction:
    a = sorted(abs(Fraction(w)) for w in ws)
    n = len(a)
    if n == 0:
        return Fraction(0)
    if variant == "absmean":
        return sum(a, Fraction(0)) / n
    if variant == "drop":
        keep = a[: n - n // 16] or a
        return sum(keep, Fraction(0)) / len(keep)
    if variant == "clip":
        p = a[max(1, math.ceil(Fraction(95, 100) * n)) - 1]
        return sum((min(x, p) for x in a), Fraction(0)) / n
    raise ValueError(variant)


def trit(w: Fraction, g: Fraction) -> int:
    if g <= 0 or 2 * abs(w) <= g:
        return 0
    return 1 if w > 0 else -1


def sse(wf, g: Fraction) -> Fraction:
    return sum(((w - g * trit(w, g)) ** 2 for w in wf), Fraction(0))


def margin(wf, g: Fraction) -> Fraction:
    h = g / 2
    return min(abs(abs(w) - h) for w in wf)


def certify(ws, variant: str) -> dict:
    wf = [Fraction(w) for w in ws]
    g = gamma_exact(ws, variant)
    if g <= 0:
        return {"variant": variant, "gamma": "0", "verdict": -1, "reason": "gamma<=0"}
    ref = [trit(w, g) for w in wf]
    s_ref = sse(wf, g)
    m = margin(wf, g)
    e = _exp10(g)
    rows, dmin = {}, {}
    for reader in ("f64", "f32"):
        dmin[reader] = None
        for d in range(1, MAX_DIGITS + 1):
            dec, _ = round_sig(g, d)
            gr = reader_value(dec, reader)
            flips = sum(1 for w, t in zip(wf, ref) if trit(w, gr) != t)
            dsse = abs(sse(wf, gr) - s_ref) / s_ref if s_ref else Fraction(0)
            rows[(reader, d)] = (flips, dsse)
        # stable: zero flips and dSSE <= RATE for d and every longer field
        for d in range(MAX_DIGITS, 0, -1):
            fl, ds = rows[(reader, d)]
            if fl or ds > Fraction(RATE_TEXT):
                break
            dmin[reader] = d
    # tensor-free certified digits: 5*10^(e-d) (+ f32 half ulp) < 2m
    cert = {}
    for reader in ("f64", "f32"):
        cert[reader] = None
        extra = half_ulp32(g) if reader == "f32" else Fraction(0)
        for d in range(1, MAX_DIGITS + 1):
            if 5 * Fraction(10) ** (e - d) + extra < 2 * m:
                cert[reader] = d
                break
    d64, d32 = dmin["f64"], dmin["f32"]
    if d64 is not None and d32 is not None and max(d64, d32) <= F32_DIGITS:
        v, d_rec = 1, max(d64, d32)
    elif d64 is not None:
        v, d_rec = 0, d64
    else:
        v, d_rec = -1, MAX_DIGITS
    _, text = round_sig(g, d_rec)
    f_rec, ds_rec = rows[("f64", d_rec)]
    if ds_rec > Fraction(RATE_TEXT):
        v = -1
    f32_at_rec = rows[("f32", d_rec)][0]
    return {
        "variant": variant, "gamma_f64": float(g).hex(), "exp10": e,
        "margin_f64": float(m).hex(), "d_min_f64": d64, "d_min_f32": d32,
        "d_cert_f64": cert["f64"], "d_cert_f32": cert["f32"],
        "d_rec": d_rec, "field": text, "flips_f64_at_rec": f_rec,
        "flips_f32_at_rec": f32_at_rec, "dsse_at_rec": float(ds_rec).hex(),
        "flips_f32_by_d": [rows[("f32", d)][0] for d in range(1, MAX_DIGITS + 1)],
        "n": len(wf), "nonzero_trits": sum(1 for t in ref if t),
        "rel_sse": float(s_ref / sum((w * w for w in wf), Fraction(0))).hex()
        if any(wf) else "0x0.0p+0",
        "verdict": v,
    }


def certify_published(ws, variant: str, text: str) -> dict:
    wf = [Fraction(w) for w in ws]
    g = gamma_exact(ws, variant)
    dec = parse_decimal(text)
    if dec is None or dec <= 0 or g <= 0:
        return {"variant": variant, "text": text, "parsed": False, "verdict": -1}
    ref = [trit(w, g) for w in wf]
    out = {"variant": variant, "text": text, "parsed": True}
    worst = 0
    for reader in ("f64", "f32"):
        gr = reader_value(dec, reader)
        out["flips_" + reader] = sum(1 for w, t in zip(wf, ref) if trit(w, gr) != t)
        s_ref = sse(wf, g)
        ds = abs(sse(wf, gr) - s_ref) / s_ref if s_ref else Fraction(0)
        out["dsse_" + reader] = float(ds).hex()
        worst = max(worst, out["flips_" + reader] + (1 if ds > Fraction(RATE_TEXT) else 0))
    out["verdict"] = 1 if worst == 0 else (0 if out["flips_f64"] == 0 else -1)
    return out


# ---- report ----------------------------------------------------------------
def field_trit(fields: dict) -> tuple[int, list]:
    notes = []
    t = 1
    bn = fields.get("JUNIOR_BITNETD")
    if bn is None:
        t, notes = min(t, 0), notes + ["JUNIOR_BITNETD absent"]
    elif not loopback_ok(bn):
        t, notes = -1, notes + ["JUNIOR_BITNETD not loopback"]
    if fields.get("JUNIOR_OFFLINE", "1") != "1":
        t, notes = -1, notes + ["JUNIOR_OFFLINE not 1"]
    if "JUNIOR_QUANT_RATE" in fields:
        r = parse_decimal(fields["JUNIOR_QUANT_RATE"])
        if r != Fraction(RATE_TEXT):
            t, notes = -1, notes + ["JUNIOR_QUANT_RATE != 0.05"]
    return t, notes


def report(ws, os_release_text: str = DEFAULT_OS_RELEASE, prev: str = GENESIS) -> dict:
    fields = parse_os_release(os_release_text)
    ft, notes = field_trit(fields)
    rows, key = [], prev
    for v in VARIANTS:
        row = certify(ws, v)
        key = close_rule(key, pack_i2s([row["verdict"]]), canon(row))
        row["key"] = key
        rows.append(row)
    published = None
    pv, pg = fields.get("JUNIOR_WINSOR_VARIANT"), fields.get("JUNIOR_WINSOR_GAMMA")
    if pv is not None or pg is not None:
        if pv in VARIANTS and pg is not None:
            published = certify_published(ws, pv, pg)
        else:
            published = {"variant": pv, "text": pg, "parsed": False, "verdict": -1}
        ft = min(ft, published["verdict"])
    green = [r for r in rows if r["verdict"] == 1]
    pick = (min(green, key=lambda r: (r["d_rec"], VARIANTS.index(r["variant"])))
            if green else None)
    verdicts = [r["verdict"] for r in rows]
    rep = {
        "os_release_id": fields.get("ID"), "field_trit": ft, "field_notes": notes,
        "published": published, "rows": rows, "verdicts": verdicts, "rate": RATE,
        "proposed_lines": ([f'JUNIOR_WINSOR_VARIANT="{pick["variant"]}"',
                            f'JUNIOR_WINSOR_GAMMA="{pick["field"]}"',
                            f'JUNIOR_QUANT_RATE="{RATE_TEXT}"'] if pick else []),
        "writes": False, "bind": BIND, "bitnetd_port": BITNETD_PORT,
        "i2sd_port": I2SD_PORT, "ports": list(PORTS), "mint": MINT,
        "socket": SOCKET, "layer_one": LAYER_ONE, "fetch": FETCH,
        "ue5_launch": UE5_LAUNCH,
    }
    rep["key"] = close_rule(key, pack_i2s(verdicts + [ft]), canon(rep))
    if pick:
        rep["proposed_lines"].append(f'JUNIOR_QUANT_KEY="{rep["key"]}"')
    return rep


# ---- synthetic f32 weights (Irwin-Hall, random.random only) -----------------
def synthetic(n: int, seed: int, outliers: bool = True):
    rng = random.Random(seed)
    ws = []
    for i in range(n):
        x = (rng.random() + rng.random() + rng.random() + rng.random() - 2.0) * 0.02
        if outliers and i % 61 == 0:
            x *= 11.0
        ws.append(f32(x))
    return ws


def read_raw_f32(path: str, limit: int) -> list:
    with open(path, "rb") as fh:
        buf = fh.read(4 * limit)
    n = len(buf) // 4
    return [w for w in struct.unpack(f"<{n}f", buf[: 4 * n]) if math.isfinite(w)]


def selftest() -> int:
    fails = 0

    def ok(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            fails += 1

    ok("flags false, loopback, five ports",
       not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
       and BIND == "127.0.0.1" and len(PORTS) == 5 and I2SD_PORT == 8767)
    f = parse_os_release(DEFAULT_OS_RELEASE + 'X_ESC="a\\"b\\\\c"\nBAD LINE\n')
    ok("os-release parse quotes and escapes",
       f["ID"] == "junioros" and f["ID_LIKE"] == "debian alpine"
       and f["X_ESC"] == 'a"b\\c' and "BAD LINE" not in f)
    ok("default os-release fields pass", field_trit(f)[0] == 1)
    ok("non-loopback bitnetd is red",
       field_trit({"JUNIOR_BITNETD": "192.168.1.5:8765"})[0] == -1)
    ok("rate 0.050 equals 0.05, 0.051 red",
       field_trit({"JUNIOR_BITNETD": "127.0.0.1:8765", "JUNIOR_QUANT_RATE": "0.050"})[0] == 1
       and field_trit({"JUNIOR_BITNETD": "127.0.0.1:8765",
                       "JUNIOR_QUANT_RATE": "0.051"})[0] == -1)
    ok("pack_i2s codes", pack_i2s([1, -1, 0, 1]) == bytes([0b01100001]))
    ok("round_sig half-even and carry",
       round_sig(Fraction(125, 100), 2)[0] == Fraction(12, 10)
       and round_sig(Fraction(135, 100), 2)[0] == Fraction(14, 10)
       and round_sig(Fraction(9999, 1000), 2) == (Fraction(10), "10"))
    ok("round_sig text parses back", parse_decimal(round_sig(Fraction(1, 3), 5)[1])
       == round_sig(Fraction(1, 3), 5)[0])
    rng = random.Random(3)
    agree64 = agree32 = True
    for _ in range(400):
        x = rng.random() * 10 ** rng.randint(-6, 3)
        s = repr(x)
        if reader_value(Fraction(s), "f64") != Fraction(float(s)):
            agree64 = False
        y = f32(x)
        if reader_value(Fraction(y), "f32") != Fraction(y):
            agree32 = False
    ok("f64 reader matches float(str)", agree64)
    ok("f32 reader fixes float32 values", agree32)
    tie = Fraction(1) + Fraction(2) ** -24
    ok("f32 reader rounds ties to even", reader_value(tie, "f32") == 1)
    rt = True
    for _ in range(200):
        v = f32(rng.random() * 3 + 1e-3)
        if reader_value(round_sig(Fraction(v), 9)[0], "f32") != Fraction(v):
            rt = False
    ok("nine sig digits round-trip 200 float32", rt)
    ok("trit boundary 2|w| > gamma", trit(Fraction(1, 2), Fraction(1)) == 0
       and trit(Fraction(-51, 100), Fraction(1)) == -1)
    ws = [0.1, -0.2, 0.3, 0.4] * 4 + [5.0]
    ok("drop gamma drops top 1/16", gamma_exact(ws, "drop")
       == sum((abs(Fraction(w)) for w in sorted(ws, key=abs)[:16]), Fraction(0)) / 16)
    ok("absmean gamma exact", gamma_exact([1.0, -3.0], "absmean") == 2)
    ok("clip gamma <= absmean", gamma_exact(ws, "clip") <= gamma_exact(ws, "absmean"))

    w = synthetic(2048, 7)
    rep = report(w)
    ok(f"report verdicts {rep['verdicts']}", len(rep["verdicts"]) == 3
       and all(v in (-1, 0, 1) for v in rep["verdicts"]))
    for r in rep["rows"]:
        d64 = r["d_min_f64"]
        ok(f"{r['variant']} f64 reaches zero flips by 17 digits (d_min {d64})",
           d64 is not None and d64 <= MAX_DIGITS)
        if r["d_cert_f64"] is not None:
            ok(f"{r['variant']} certified digits are sufficient",
               d64 is not None and d64 <= r["d_cert_f64"])
        if r["verdict"] == 1:
            ok(f"{r['variant']} green row has zero flips at d_rec",
               r["flips_f64_at_rec"] == 0 and r["flips_f32_at_rec"] == 0)
    one = certify(w, "absmean")
    ok("one digit flips some trits (gamma moves up to 5%)",
       sum(one["flips_f32_by_d"][:1]) > 0 or one["d_min_f32"] == 1)
    # adversarial weight with 2|w| strictly between exact gamma and its f32
    # reading (fixed point, gamma includes the weight): no stable f32 field.
    hard, x = list(w), 0.0
    for _ in range(40):
        gh = gamma_exact(hard + [x], "absmean")
        x = float((gh + reader_value(gh, "f32")) / 4)
    hard.append(x)
    gh = gamma_exact(hard, "absmean")
    g32 = reader_value(gh, "f32")
    lo, hi = sorted((gh, g32))
    placed = lo < 2 * Fraction(x) < hi
    rh = certify(hard, "absmean")
    ok(f"adversarial weight placed {placed}: f32 not stable, verdict 0",
       placed and rh["d_min_f32"] is None and rh["verdict"] == 0
       and rh["d_min_f64"] is not None and rh["d_cert_f32"] is None)
    pub = report(w, DEFAULT_OS_RELEASE + 'JUNIOR_WINSOR_VARIANT="absmean"\n'
                 'JUNIOR_WINSOR_GAMMA="1e-1"\n')
    ok("stale published gamma is red", pub["published"]["verdict"] == -1
       and pub["field_trit"] == -1)
    good = rep["rows"][0]["field"]
    pub2 = report(w, DEFAULT_OS_RELEASE + 'JUNIOR_WINSOR_VARIANT="absmean"\n'
                  f'JUNIOR_WINSOR_GAMMA="{good}"\n')
    ok("certified published gamma not red", pub2["published"]["verdict"] >= 0)
    bad = report(w, DEFAULT_OS_RELEASE + 'JUNIOR_WINSOR_VARIANT="median"\n')
    ok("unknown variant is red", bad["field_trit"] == -1)
    r1, r2, r3 = report(w), report(w), report(w, prev="f" * 16)
    ok("key deterministic, 16 hex", r1["key"] == r2["key"] and len(r1["key"]) == 16
       and int(r1["key"], 16) >= 0)
    ok("key depends on prev", r1["key"] != r3["key"])
    ok("report JSON canonical", json.loads(canon(r1))["key"] == r1["key"])
    ok("proposed lines are printed, never written",
       r1["writes"] is False and all(l.startswith("JUNIOR_") for l in r1["proposed_lines"]))
    ok("zero tensor is red", certify([0.0] * 8, "absmean")["verdict"] == -1)
    print(f"selftest: {'PASS' if fails == 0 else 'FAIL'} ({fails} failed)")
    return 0 if fails == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--os-release", help="read-only path to an os-release file")
    ap.add_argument("--weights", help="local raw little-endian f32 file (no download)")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    text = DEFAULT_OS_RELEASE
    if a.os_release:
        with open(a.os_release, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    ws = read_raw_f32(a.weights, a.n) if a.weights else synthetic(a.n, a.seed)
    rep = report(ws, text, a.prev)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if rep["field_trit"] >= 0 and 1 in rep["verdicts"] else 1


if __name__ == "__main__":
    sys.exit(main())
