"""Judges what a recorder heard on the Mac CI's loopback devices (tests/coreaudio).

BlackHole hands back what was played at sample time S as input at sample time S, on the same host
time line, so the recorder's own ADC stamps (through the same DLL the engine uses) tell when every
recorded frame was played - absolutely, with no latency of BlackHole's own (stock BlackHole has
none; the twins report theirs as stream latency, which the player adds and the judge takes back).

Two kinds of content are judged:
- engine_play's (content()): a 997 Hz sine at -6 dBFS on the left, for continuity, and on the
  right the code ((n mod 32768) - 16384) / 32768, which says which sample n is playing to a small
  fraction of a sample (and survives the 16-bit wire exactly, for the listener in P8b).
- `syncaudio host --click --tone`: a 997 Hz sine at -12 dBFS on the left and a click every 0.5 s
  on the right, timed against the host's own stamps of every block (--report).

A hole is 8 or more frames of silence on both channels where sound is due; a discontinuity is a
frame that isn't the content it should be (a jump in the code, or not the sine for its sample: a
click, a fade, stale audio). Bad frames closer than 10 ms are one event.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(here)))
import audio_engine as ae  # noqa: E402
from syncaudio import ClickSource, make_click  # noqa: E402

RATE = 48000
SINE_HZ = 997.0
LEVEL = 10 ** (-6 / 20)
CODE = 32768
WRAP = 32           # frames each side of the code's wrap, which the resampler smears
HOLE = 8            # silent frames in a row that make a hole
QUIET = 1e-4
JUMP = 0.05         # samples: the code may step by 1 +- this (speed nudges are <= 0.003)
WAVE = 3e-3         # how far the left channel may be from the sine for its sample
MERGE = 480         # bad frames closer than this are one event
BEAT = RATE // 2    # ClickSource: a click every 0.5 s
WIN = 240           # frames per window when the content's exact sample isn't known (clicks)


def content(n0: int, k: int) -> np.ndarray:
    """engine_play's audio for samples n0 ... n0 + k."""
    n = np.arange(n0, n0 + k)
    pcm = np.empty((k, 2), np.float32)
    pcm[:, 0] = LEVEL * np.sin(2 * np.pi * SINE_HZ * n / RATE)
    pcm[:, 1] = (n % CODE - CODE // 2) / CODE
    return pcm


def wrapped(x):
    """x folded into [-CODE/2, CODE/2): how far apart two code readings are."""
    return (np.asarray(x) + CODE // 2) % CODE - CODE // 2


# --------------------------------------------------------------------------------- recordings
def frame_times(recs: np.ndarray, n: int, rate: float = RATE) -> np.ndarray:
    """Host time of recorded frames 0 ... n-1: the recorder's ADC stamps through the engine's DLL,
    each frame on the line as it stood after the callback that captured it."""
    clock = ae.DeviceClock(rate)
    pos, lines = [], []
    for p, t, f, cur in zip(recs["pos"].tolist(), recs["dev_time"].tolist(), recs["frames"].tolist(),
                            recs["current_time"].tolist()):
        clock.add(p, t, f, cur)
        if clock.t0 is not None:
            pos.append(p)
            lines.append((clock.t0, clock.p0, clock.per))
    if not lines:
        raise ValueError("the recorder has no usable timestamps")
    i = np.arange(n)
    j = np.maximum(np.searchsorted(np.asarray(pos), i, side="right") - 1, 0)
    t0, p0, per = (np.asarray(v)[j] for v in zip(*lines))
    return t0 + (i - p0) * per


class Recording:
    """recorder.py's files: PREFIX.pcm.npy, PREFIX.recs.npy, PREFIX.rec.json."""

    def __init__(self, prefix: str, pcm=None, recs=None, meta=None):
        self.pcm = np.load(prefix + ".pcm.npy") if pcm is None else pcm
        self.recs = np.load(prefix + ".recs.npy") if recs is None else recs
        if meta is None:
            with open(prefix + ".rec.json") as f:
                meta = json.load(f)
        self.meta = meta
        self.t = frame_times(self.recs, len(self.pcm))

    def window(self, t_from=-math.inf, t_to=math.inf):
        sel = np.nonzero((self.t >= t_from) & (self.t < t_to))[0]
        return self.t[sel], self.pcm[sel].astype(np.float64)

    def sound(self):
        """(first, last) host time with any sound, or None."""
        loud = np.nonzero(np.any(np.abs(self.pcm) >= QUIET, axis=1))[0]
        return (float(self.t[loud[0]]), float(self.t[loud[-1]])) if len(loud) else None

    def rms(self, t_from=-math.inf, t_to=math.inf) -> float:
        _, y = self.window(t_from, t_to)
        return float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0


# ----------------------------------------------------------------------------------- analysis
def _runs(mask: np.ndarray, min_len: int = 1) -> np.ndarray:
    """Frames inside runs of at least min_len True values."""
    if not mask.any():
        return mask
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    out = np.zeros(len(mask), bool)
    for s, e in zip(starts, ends):
        if e - s >= min_len:
            out[s:e] = True
    return out


def events(bad: np.ndarray, hole: np.ndarray, t: np.ndarray, frame_s: float = 1 / RATE) -> list:
    """Bad frames closer than MERGE as one event each: when, how long, and a hole (silence in it) or
    a discontinuity."""
    where = np.nonzero(bad)[0]
    if not len(where):
        return []
    cut = np.nonzero(np.diff(where) > MERGE)[0]
    out = []
    for a, b in zip(np.r_[0, cut + 1], np.r_[cut, len(where) - 1]):
        lo, hi = where[a], where[b]
        out.append({"t": round(float(t[lo]), 4), "ms": round((hi - lo + 1) * frame_s * 1000, 2),
                    "kind": "hole" if hole[lo:hi + 1].any() else "discontinuity"})
    return out


def analyze(y: np.ndarray, n_ref: np.ndarray, wave: float = WAVE, jump: float = JUMP) -> dict:
    """Frame by frame, for engine_play's content: which sample each frame is (n_dec, the code read
    next to n_ref, the sample expected or a rough guess), and what is wrong with it."""
    silent = (np.abs(y[:, 0]) < QUIET) & (np.abs(y[:, 1]) < QUIET)
    hole = _runs(silent, HOLE)
    m = y[:, 1] * CODE + CODE // 2
    n_dec = n_ref + wrapped(m - n_ref)
    # Where the code wraps, the resampler smears its jump over +-16 frames into values that read as
    # any sample: find those frames from where the content is (n_ref + the typical offset per 1024).
    typical = np.zeros(len(y))
    for i in range(0, len(y), 1024):
        typical[i:i + 1024] = np.median((n_dec - n_ref)[i:i + 1024])
    at = (n_ref + typical) % CODE
    wrap = (at < WRAP) | (at > CODE - WRAP)
    off = np.abs(y[:, 0] - LEVEL * np.sin(2 * np.pi * SINE_HZ * n_dec / RATE)) > wave
    step = np.zeros(len(y), bool)
    step[1:] = (np.abs(np.diff(n_dec) - 1) > jump) & ~wrap[1:] & ~wrap[:-1]
    bad = hole | (~wrap & ~silent & (off | step))
    return {"n_dec": n_dec, "hole": hole, "bad": bad, "good": ~bad & ~wrap & ~silent}


def _summary(ev: list, bad: np.ndarray, t: np.ndarray) -> dict:
    return {"judged_s": round(len(t) / RATE, 1), "holes": sum(e["kind"] == "hole" for e in ev),
            "discontinuities": sum(e["kind"] == "discontinuity" for e in ev),
            "lost_ms": round(float(bad.sum()) / RATE * 1000, 2), "events": ev[:20]}


def _spread(err: np.ndarray, t: np.ndarray) -> dict:
    """err (s, + = late) at times t: median, p1, p99, max |err|, and drift (the median over the last
    10 s minus over the first 10 s; thirds of a window shorter than 30 s), in ms."""
    if not len(err):
        return {"err_median_ms": None, "err_p1_ms": None, "err_p99_ms": None, "err_max_ms": None, "drift_ms": None}
    ms = err * 1000
    edge = min(10.0, (t[-1] - t[0]) / 3)
    first, last = ms[t <= t[0] + edge], ms[t >= t[-1] - edge]
    return {"err_median_ms": round(float(np.median(ms)), 4), "err_p1_ms": round(float(np.percentile(ms, 1)), 4),
            "err_p99_ms": round(float(np.percentile(ms, 99)), 4), "err_max_ms": round(float(np.abs(ms).max()), 4),
            "drift_ms": round(float(np.median(last) - np.median(first)), 4)}


def ramp(rec: Recording, T0: float, D: float, strlat: float, t_from=-math.inf, t_to=math.inf) -> dict:
    """engine_play's content, stamped play_at(n) = T0 + n / RATE + D, on a device whose stream latency
    is strlat s: err = when a frame was played - (play_at(n) - strlat)."""
    t, y = rec.window(t_from, t_to)
    if not len(t):
        return {"judged_s": 0.0, "error": "nothing recorded in that window"}
    n_exp = (t + strlat - T0 - D) * RATE
    a = analyze(y, n_exp)
    good = a["good"]
    err = (n_exp[good] - a["n_dec"][good]) / RATE
    ev = events(a["bad"], a["hole"], t)
    sine = np.sin(2 * np.pi * SINE_HZ * a["n_dec"][good] / RATE)
    gain = float(np.dot(y[good, 0], sine) / np.dot(sine, sine)) / LEVEL if good.any() else None
    return {**_summary(ev, a["bad"], t), **_spread(err, t[good]),
            "gain": None if gain is None else round(gain, 5), "good_s": round(float(good.sum()) / RATE, 1)}


def _tone_windows(y, level, phase_tol):
    """The left channel's 997 Hz tone, window by window: bad where its level or phase isn't steady
    (a jump, a fade, a gap), measured without knowing which sample each frame is. The phase may drift
    (a speed nudge, a delay glide: up to 0.1 rad a window); it may not jump against its neighbours
    by more than phase_tol (a whole sample is 0.13 rad)."""
    k = len(y) // WIN
    if k < 3:
        return np.zeros(len(y), bool)
    w = 2 * np.pi * SINE_HZ / RATE
    seg = y[:k * WIN, 0].reshape(k, WIN)
    z = seg @ np.exp(-1j * w * np.arange(WIN)) * 2 / WIN
    amp = np.abs(z) / level
    step = np.angle(z[1:] * np.conj(z[:-1]) * np.exp(-1j * w * WIN))     # phase advance beyond a pure tone
    pad = np.pad(step, 10, mode="edge")
    ref = np.median(np.lib.stride_tricks.sliding_window_view(pad, 21), axis=1)     # the last and next 50 ms
    off = np.abs(np.angle(np.exp(1j * (step - ref))))
    bad_w = (np.abs(amp - 1) > 0.1)
    bad_w[1:] |= off > phase_tol
    bad_w[:-1] |= off > phase_tol
    bad = np.zeros(len(y), bool)
    bad[:k * WIN] = np.repeat(bad_w, WIN)
    return bad


def clicks(rec: Recording, stamps: np.ndarray, strlat: float, t_from=-math.inf, t_to=math.inf,
           opus: bool = False) -> dict:
    """`syncaudio host --click --tone` played on a device of stream latency strlat: each click's
    onset (matched to its shape, to a fraction of a sample) against the host's stamp for it, and the
    tone's continuity. stamps: [[n, play_at], ...] of every block, from the host's --report. Opus
    wobbles the tone's phase by up to 0.07 rad a window (PCM: 0.0003), so it is allowed 0.1."""
    t, y = rec.window(t_from, t_to)
    if not len(t):
        return {"judged_s": 0.0, "error": "nothing recorded in that window"}
    silent = (np.abs(y[:, 0]) < QUIET) & (np.abs(y[:, 1]) < QUIET)
    hole = _runs(silent, HOLE)
    bad = hole | _tone_windows(y, ClickSource.TONE_LEVEL, 0.1 if opus else 0.03)
    stamp = {int(n): float(p) for n, p in stamps}
    shapes = {f: make_click(RATE, f).astype(np.float64) for f in (2000, 1000)}
    err, when, missing = [], [], 0
    span = int(0.2 * RATE)
    for n in sorted(k for k in stamp if k % BEAT == 0):
        due = stamp[n] - strlat
        if not t[0] + 0.25 <= due <= t[-1] - 0.25:
            continue
        shape = shapes[2000 if (n // BEAT) % 4 == 0 else 1000]
        i0 = int(np.searchsorted(t, due)) - span
        seg = y[max(0, i0):i0 + 2 * span + len(shape), 1]
        i0 = max(0, i0)
        size = 1 << int(math.ceil(math.log2(len(seg) + len(shape))))
        c = np.fft.irfft(np.fft.rfft(seg, size) * np.conj(np.fft.rfft(shape, size)), size)[:len(seg) - len(shape) + 1]
        j = int(np.argmax(c))
        energy = math.sqrt(float(np.dot(seg[j:j + len(shape)], seg[j:j + len(shape)])) * float(np.dot(shape, shape)))
        if energy <= 0 or c[j] / energy < 0.8:
            missing += 1
            continue
        frac = 0.0
        if 0 < j < len(c) - 1:      # the peak between frames (a parabola through three)
            a, b, d = c[j - 1], c[j], c[j + 1]
            frac = 0.5 * (a - d) / (a - 2 * b + d) if a - 2 * b + d else 0.0
        f = i0 + j + frac
        played = np.interp(f, np.arange(len(t)), t)
        err.append(played - due)
        when.append(played)
    ev = events(bad, hole, t)
    err, when = np.asarray(err), np.asarray(when)
    return {**_summary(ev, bad, t), **_spread(err, when), "clicks": len(err), "clicks_missing": missing}


def relative(y: np.ndarray) -> np.ndarray:
    """A rough sample index for every frame of a recording whose timing isn't known in advance (a
    listener's): the code says n mod 32768, and n - frame stays about constant from one 1024 frames
    to the next (it moves only with the speed, a jump or a delay change), so it is followed chunk by
    chunk, never across the code's wrap and never through silence. The whole number of code periods
    is then the one that makes the left channel the sine for n: 997 Hz x 32768 / 48000 isn't a whole
    number of cycles, so only one in every 375 periods fits."""
    m = y[:, 1] * CODE + CODE // 2
    i = np.arange(len(y))
    loud = ~((np.abs(y[:, 0]) < QUIET) & (np.abs(y[:, 1]) < QUIET))
    if not loud.any():
        return i.astype(np.float64)
    first = int(np.argmax(loud))
    c, rough = float(wrapped(m[first] - first)), np.zeros(len(y))
    for s in range(0, len(y), 1024):
        part = slice(s, s + 1024)
        if loud[part].mean() > 0.5:
            c += float(np.median(wrapped(m[part] - i[part] - c)))
        rough[part] = c
    n = i + rough
    j = np.nonzero(loud)[0][:4800]
    k = min(range(375), key=lambda k: float(np.mean(np.abs(
        y[j, 0] - LEVEL * np.sin(2 * np.pi * SINE_HZ * (n[j] + k * CODE) / RATE)))))
    return n + k * CODE


def between(a: Recording, strlat_a: float, b: Recording, strlat_b: float, t_from=-math.inf, t_to=math.inf,
            b_wire: bool = False) -> dict:
    """Two devices playing engine_play's content (through a host and a listener): how far apart they
    are at the ear, sample by sample (+ = a plays later), and each one's own holes and jumps.
    b_wire: b's audio crossed the 16-bit wire, which rounds the code to a whole sample while the
    sine keeps the fraction the tone was played at (so the two may disagree by half a sample)."""
    out, sides = {}, []
    for name, rec, strlat in (("a", a, strlat_a), ("b", b, strlat_b)):
        t, y = rec.window(t_from - strlat, t_to - strlat)
        if not len(t):
            return {"error": f"nothing recorded on {name} in that window"}
        res = analyze(y, relative(y), *((0.05, 1.5) if name == "b" and b_wire else ()))
        ev = events(res["bad"], res["hole"], t)
        out[name] = _summary(ev, res["bad"], t)
        sides.append((t + strlat, res["n_dec"], res["good"]))
    (ta, na, ga), (tb, nb, gb) = sides
    fb = np.interp(ta, tb, np.arange(len(tb)), left=np.nan, right=np.nan)
    ok = ga & ~np.isnan(fb)
    ib = np.clip(np.floor(np.nan_to_num(fb)).astype(np.int64), 0, len(tb) - 2)
    ok &= gb[ib] & gb[ib + 1]
    nb_at = nb[ib] + (np.nan_to_num(fb) - ib) * (nb[ib + 1] - nb[ib])
    diff = wrapped(nb_at[ok] - na[ok]) / RATE      # b ahead in the content = a later
    out.update(_spread(diff, ta[ok]))
    out["compared_s"] = round(float(ok.sum()) / RATE, 1)
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="judge a recording of engine_play's content")
    ap.add_argument("prefix", help="the recorder's --out")
    ap.add_argument("--stats", required=True, help="engine_play's PREFIX.json")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0, help="s after the engine started")
    ap.add_argument("--to", dest="t_to", type=float, default=math.inf, help="s after the engine started")
    args = ap.parse_args(argv)
    with open(args.stats) as f:
        st = json.load(f)
    r = ramp(Recording(args.prefix), st["T0"], st["D"], st["strlat"], st["started"] + args.t_from,
             st["started"] + args.t_to)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
