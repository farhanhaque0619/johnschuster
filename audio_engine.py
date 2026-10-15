"""SyncAudio 2.0's audio engine: everything between libsacore's rings and the rest of the app.

CoreAudio's IO threads only ever run C (native/sacore.c). It copies between the device and a ring
whose positions are DEVICE FRAME INDICES, so index -> time stays one straight line even when macOS
skips IO cycles. Everything here runs on ordinary Python threads, ahead of the device:

    DeviceClock     per-callback (index, DAC time) records -> that straight line (a filtered DLL)
    StreamBuffer    received audio by sample index (kept in C) and when each sample is due
    OutputEngine    the producer: for ring index w, write the audio due when w reaches the ear,
                    150 ms - 1 s before it plays (one GIL-free C resample per block)
    CaptureEngine   drains the capture ring (BlackHole) in 10 ms blocks, stamped by its own DLL
    AudioOutput / AudioInput   open the streams so PortAudio can't pick a tiny buffer, watch them,
                    and say when to reopen them (the self-heal ladder)
    SelfTest        "Test my headphones": the 1.1.6 way, then this way, and a verdict
    GcTimer / GilProbe   how long Python itself stalled (diagnostics)

Every class takes `now` (and the sounddevice module) from its caller, so the tests run all of it in
virtual time against the real C callbacks (tests/vdevice.py).
"""
from __future__ import annotations

import bisect
import gc
import math
import statistics
import sys
import threading
import time
import traceback
import types
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

import sacore

OUT_BLOCK, OUT_BLOCK_BT, OUT_BLOCK_MAX, IN_BLOCK = 1024, 2048, 4096, 512
BT_FIXED_S = 0.060          # fixed latency at/above this = Bluetooth-class (AirPods report ~0.16 s)
MIN_OK_FRAMES = 512
PA_LATENCY = 0.0            # with an explicit blocksize: host buffer == blocksize (pa_mac_core.c:1652-1671)
ENGINE = "c-ring/2.0"
STOP_WAIT = 2.0             # s an engine's stop() waits for its thread (a tick is short: this one is starved)


# ------------------------------------------------------------------------------- the IO plan
@dataclass
class IOPlan:
    blocksize: int
    latency: float
    level: int
    bt_class: bool
    fixed_s: float | None
    source: str             # where fixed_s came from: "coreaudio" | "portaudio" | "none"
    reason: str = ""


def fixed_latency(sd, index: int, kind: str, facts) -> tuple:
    """(seconds, source): the latency a device adds on its own, whatever its buffer size.

    PortAudio sizes the IO buffer as (requested latency - this), so for AirPods (~160 ms) 1.1.6's
    latency=0.1 asked for less than nothing and got the device minimum (~1 ms): the static."""
    if facts and facts.get("nominal_rate", 0) > 0:
        frames = facts["latency"] + facts["safety"] + facts["stream_latency"]
        return frames / facts["nominal_rate"], "coreaudio"
    try:
        info = sd.query_devices(index)
        # PortAudio's own default_low = (fixed + max(64, smallest buffer)) / rate (pa_mac_core.c:545-568)
        low = info[f"default_low_{kind}_latency"] - 64 / info["default_samplerate"]
        return max(0.0, low), "portaudio"
    except Exception:       # noqa: BLE001 - an odd device must still open
        return None, "none"


def plan_output(sd, index: int, rate: float, facts, level: int = 0) -> IOPlan:
    """How to open an output: the blocksize to ask for, and the latency to pass with it.

    Level 0 asks for a buffer outright (2048 frames for Bluetooth, 1024 otherwise) with latency 0,
    which makes PortAudio's host buffer exactly that. Level 1 doubles it. Level 2 lets PortAudio and
    macOS choose (for devices that won't play at our size); the C callback takes any frame count."""
    fixed, source = fixed_latency(sd, index, "output", facts)
    transport = (facts or {}).get("transport")
    bt = transport in ("blue", "blea") or (fixed is not None and fixed >= BT_FIXED_S)
    base = OUT_BLOCK_BT if bt else OUT_BLOCK
    level = max(0, min(2, int(level)))
    if level == 0:
        blocksize, latency = base, PA_LATENCY
    elif level == 1:
        blocksize, latency = min(2 * base, OUT_BLOCK_MAX), PA_LATENCY
    else:       # 50 ms of buffer on top of the fixed part; if that's unknown, plenty
        blocksize, latency = 0, (fixed + 0.05) if fixed is not None else 0.25
    return IOPlan(blocksize, latency, level, bt, fixed, source)


# ------------------------------------------------------------------------------ device clock
class DeviceClock:
    """When device frame p reaches the ear (output) or was sampled (input), in host time.

    PortAudio stamps every callback with macOS's time for its first frame (the DAC / ADC time). The
    stamps wobble by a few ms, Bluetooth most, so they go through a second-order delay-locked loop.
    It starts wide (0.5 Hz) and narrows as 1/age down to 0.02 Hz: like a straight-line fit through
    everything seen so far, so the rate it learns while wide isn't the wobble's (a quicker narrowing
    keeps a rate that is ~1000 ppm off, and the line then drifts ms away for half a minute). The
    wobble never reaches the music; the device's real rate is followed. One wild stamp is ignored;
    three in a row mean the timeline really moved (a restart, a latency change): the line starts
    again there and settles afresh.

    IO cycles macOS skipped don't show here: the C callback already moved the index past them. Losses
    too small for it to see (under half a buffer) make the device look slow instead: a rate more than
    1000 ppm off for 5 s is "lossy" - reported, and followed with a wider loop, which then takes a
    loss of up to half a buffer as rate rather than as a re-anchor. Losses over REANCHOR re-anchor
    the line, exactly like a latency change; three of those forward in 30 s are no latency change
    either. Lossy ends once the stamps' own slope over 30 s is back within 500 ppm."""

    BW_START, BW_END, BW_LOSSY = 0.5, 0.02, 0.3         # Hz
    CLIP, REANCHOR = 0.002, 0.010
    PER_CLAMP, PER_CLAMP_LOSSY = 0.01, 0.03
    LOSSY_AGE = 5.0         # s of settling before a rate counts as loss
    LOSSY_ON, LOSSY_ON_S, LOSSY_OFF, LOSSY_OFF_S = 1000.0, 5.0, 500.0, 30.0
    LOSSY_JUMPS = 3         # forward re-anchors within LOSSY_OFF_S

    def __init__(self, rate: float, offset: float = 0.0, lead_guess: float | None = None):
        self.nominal = rate
        self.offset = offset            # what the stamps leave out (kAudioStreamPropertyLatency)
        self.per = 1.0 / rate           # host seconds per device frame
        self.t0 = None                  # the line: frame p0 at host time t0
        self.p0 = 0
        self.started = None             # when the line was (re)started: the loop settles from here
        self.records = self.accepted = self.invalid = 0
        self.reanchors = self.outliers = 0
        self.lossy = False
        self._odd = 0
        self._high = None               # since when the loop's rate has been past LOSSY_ON
        self._jumps: list = []          # when the line last re-anchored forward
        self._fit: deque = deque()      # lossy: (pos, t) accepted in the last LOSSY_OFF_S
        self._lossy_t = self._fit_t = None
        self.lead_guess = lead_guess    # stamp - "now", before any stamp was seen (stream.latency)
        self.leads: deque = deque()     # (current_time, stamp - current_time), the last 2 s
        self.lead_min = self.lead_max = None
        self.first_lead = None          # their median over the first 2 s (diagnostics: probe_d_ms)
        self.frames_min = self.frames_max = None
        self.res_sq, self.res_n, self.res_max = 0.0, 0, 0.0

    @property
    def ready(self) -> bool:
        return self.accepted >= 3

    @property
    def rate_ppm(self) -> float:
        """+: the device delivers fewer frames per second than nominal (or loses time)."""
        return (self.per * self.nominal - 1) * 1e6

    @property
    def lead(self):
        """Median of (stamp - PortAudio's now) over the last 2 s: IOProc to DAC."""
        leads = tuple(self.leads)       # (one C-level copy: the engine thread may be appending)
        return statistics.median(v for _, v in leads) if leads else self.lead_guess

    def feed(self, recs) -> None:
        for pos, t, frames, cur in zip(recs["pos"].tolist(), recs["dev_time"].tolist(),
                                       recs["frames"].tolist(), recs["current_time"].tolist()):
            self.add(pos, t, frames, cur)

    def add(self, pos: int, t: float, frames: int, cur: float) -> None:
        self.records += 1
        self.frames_min = frames if self.frames_min is None else min(self.frames_min, frames)
        self.frames_max = frames if self.frames_max is None else max(self.frames_max, frames)
        if t > 0:
            lead = t - cur
            self.leads.append((cur, lead))
            if self.first_lead is None and cur - self.leads[0][0] >= 2.0:
                self.first_lead = self.lead
            while self.leads[0][0] < cur - 2.0:
                self.leads.popleft()
            self.lead_min = lead if self.lead_min is None else min(self.lead_min, lead)
            self.lead_max = lead if self.lead_max is None else max(self.lead_max, lead)
        else:           # PortAudio had no stamp for this buffer: "now" plus the usual lead
            self.invalid += 1
            if self.lead is None or cur <= 0:
                return
            t = cur + self.lead
        if self.t0 is None:
            self.t0, self.p0, self.started, self.accepted = t, pos, t, 1
            return
        pred = self.t0 + (pos - self.p0) * self.per
        e = t - pred
        # (lossy: anything the C side doesn't count as a skip is followed, as rate)
        if abs(e) > (max(self.REANCHOR, 0.5 * frames / self.nominal) if self.lossy else self.REANCHOR):
            self.outliers += 1
            self._odd += 1
            if self._odd >= 3:          # three in a row: the timeline really moved
                self.t0, self.p0, self.started, self._odd = t, pos, t, 0
                self.reanchors += 1
                if e > 0:
                    self._jumped(t)
            return
        self._odd = 0
        age = t - self.started
        bw = self.BW_LOSSY if self.lossy else max(self.BW_END, self.BW_START / max(1.0, age))
        w = 2 * math.pi * bw * frames * self.per
        ec = max(-self.CLIP, min(self.CLIP, e))
        self.t0, self.p0 = pred + math.sqrt(2) * w * ec, pos
        clamp = self.PER_CLAMP_LOSSY if self.lossy else self.PER_CLAMP
        self.per = min((1 + clamp) / self.nominal, max((1 - clamp) / self.nominal, self.per + w * w * ec / frames))
        self.accepted += 1
        self._check_lossy(pos, t, age)
        if age > 2:
            self.res_sq += e * e
            self.res_n += 1
            self.res_max = max(self.res_max, abs(e))

    def _jumped(self, t: float) -> None:
        """The line moved forward by more than REANCHOR and less than the C side sees as a skip: a
        latency change, or macOS losing a slice. Again and again, it is the losses."""
        self._jumps = [j for j in self._jumps if j > t - self.LOSSY_OFF_S] + [t]
        if not self.lossy and len(self._jumps) >= self.LOSSY_JUMPS:
            self._set_lossy(True, t)

    def _check_lossy(self, pos: int, t: float, age: float) -> None:
        if not self.lossy:
            self._high = (self._high or t) if abs(self.rate_ppm) > self.LOSSY_ON else None
            if age > self.LOSSY_AGE and self._high is not None and t - self._high >= self.LOSSY_ON_S:
                self._set_lossy(True, t)
            return
        # Out again once the stamps of the last 30 s lie on a line within 500 ppm of nominal. Not on the
        # wide loop's rate: that swings with Bluetooth's wobble by thousands of ppm, for ever.
        self._fit.append((pos, t))
        while self._fit[0][1] < t - self.LOSSY_OFF_S:
            self._fit.popleft()
        if t - self._fit_t >= 1.0:
            self._fit_t = t
            p, tt = np.array(self._fit).T
            x = p - p.mean()
            per = (x * (tt - tt.mean())).sum() / (x * x).sum()
            if t - self._lossy_t >= self.LOSSY_OFF_S and abs(per * self.nominal - 1) * 1e6 < self.LOSSY_OFF:
                self.per = per          # (the narrow loop carries on from the real rate, not the wobble's)
                self._set_lossy(False, t)

    def _set_lossy(self, on: bool, t: float) -> None:
        self.lossy, self._lossy_t, self._fit_t = on, t, t
        self._high, self._jumps = None, []
        self._fit.clear()

    def time_of(self, pos: float) -> float:
        return self.t0 + (pos - self.p0) * self.per + self.offset

    @property
    def loss_pct(self) -> float:
        return max(0.0, self.rate_ppm) / 1e4 if self.lossy else 0.0

    def stats(self) -> dict:
        lead = self.lead
        return {"frames_min": self.frames_min, "frames_max": self.frames_max,
                "lead_ms": [round(v * 1000, 1) for v in (self.lead_min, lead, self.lead_max)]
                if self.lead_min is not None else None,
                "rate_ppm": round(self.rate_ppm, 1),
                "jitter_ms": round(math.sqrt(self.res_sq / self.res_n) * 1000, 3) if self.res_n else None,
                "jitter_max_ms": round(self.res_max * 1000, 2), "reanchors": self.reanchors,
                "outliers": self.outliers, "invalid": self.invalid, "lossy": self.lossy,
                "loss_pct": round(self.loss_pct, 2)}


# ----------------------------------------------------------------------------- stream buffer
class StreamBuffer:
    """Received audio by stream sample index n (in C: sacore.Content), and when each n is due.

    The host stamps every block with play_at on its own clock. Every block's stamp is kept (an
    anchor), so a delay that glides plays exactly as stamped, not as a guess from the first block.
    A stamp more than STEP off the line of the blocks before it starts a new segment AT THAT SAMPLE
    (a delay change): every Mac switches at the same sample, however late the block arrived.
    Only the network (or capture) thread feeds it and only the output engine reads it; no IO thread
    touches the lock."""

    STEP = 0.001
    KEEP = 11.0         # seconds of anchors kept: a little more than the content ring holds

    def __init__(self, rate: int, channels: int, seconds: float = 8.0):
        self.rate, self.channels = rate, channels
        self.content = sacore.Content(channels, rate, seconds)
        self.lock = threading.Lock()
        self.an: list = []      # each block's first sample index...
        self.at: list = []      # ...and its play_at
        self.seg: list = []     # where each segment starts (always an anchor too)
        self.holes = 0          # samples that never arrived (played as silence)
        self.last_n = None      # end of the newest block
        self.base = 0           # added to n: a new stream that goes back in n moves past the old one
        self.readers: list = []     # the output engines that play it (OutputEngine adds itself)

    def feed(self, n: int, play_at: float, pcm: np.ndarray) -> None:
        with self.lock:
            n += self.base
            if self.last_n is not None and n < self.last_n:     # a new stream, back in n: put it far past
                self.base += self.last_n - n + 2 * self.content.cap     # the old one, so nothing old plays
                n = self.last_n + 2 * self.content.cap
            if self.last_n is None or n - self.last_n > self.content.cap:
                self.an, self.at, self.seg = [n], [play_at], [n]
            else:
                j = len(self.an) - 1
                pred = self.at[j] + (n - self.an[j]) * self._slope(j)
                if abs(play_at - pred) > self.STEP:     # the delay changed from this sample on
                    self.seg.append(n)
                self.an.append(n)
                self.at.append(play_at)
                if len(self.an) > 1.1 * self.KEEP * self.rate / max(1, len(pcm)):
                    self._prune()
            self.last_n = n + len(pcm)
        # after the anchor: whoever sees these samples can also see when they are due
        self.holes += self.content.put(n, pcm)

    def _prune(self):
        i = bisect.bisect_left(self.an, self.last_n - self.KEEP * self.rate)
        del self.an[:i], self.at[:i]
        self.seg = [s for s in self.seg if s > self.an[0]]
        self.seg.insert(0, self.an[0])      # what's left of the oldest segment starts at the oldest anchor

    # The lookups below run under the lock; i is an anchor index, lo/hi bound its segment's anchors.
    def _bounds(self, j: int):
        lo = bisect.bisect_left(self.an, self.seg[j])
        hi = bisect.bisect_left(self.an, self.seg[j + 1]) if j + 1 < len(self.seg) else len(self.an)
        return lo, hi

    def _slope(self, i: int, lo: int | None = None, hi: int | None = None) -> float:
        """Seconds per sample around anchor i, within its segment."""
        if lo is None:
            lo, hi = self._bounds(max(0, bisect.bisect_right(self.seg, self.an[i]) - 1))
        if i + 1 < hi:
            return (self.at[i + 1] - self.at[i]) / (self.an[i + 1] - self.an[i])
        if i > lo:
            return (self.at[i] - self.at[i - 1]) / (self.an[i] - self.an[i - 1])
        return 1.0 / self.rate

    def _locate(self, x: float):
        """(anchor at or before x in x's segment, seconds per sample there)."""
        j = max(0, bisect.bisect_right(self.seg, x) - 1)
        lo, hi = self._bounds(j)
        i = max(lo, bisect.bisect_right(self.an, x, lo, hi) - 1)
        return i, self._slope(i, lo, hi)

    def play_at(self, x: float):
        """Host time sample x is due (between anchors: on the line between them)."""
        with self.lock:
            if not self.an:
                return None
            i, slope = self._locate(x)
            return self.at[i] + (x - self.an[i]) * slope

    def slope(self, x: float):
        """d(play_at)/dn at x: 1/rate, or (1 + g)/rate while the host glides the delay by g."""
        with self.lock:
            return self._locate(x)[1] if self.an else None

    def index_at(self, t: float):
        """(sample due at host time t, in_gap). in_gap: t falls in the silence a delay increase put
        before segment start n_s, and n_s is returned."""
        with self.lock:
            if not self.an:
                return None, False
            for j in range(len(self.seg) - 1, -1, -1):
                lo, hi = self._bounds(j)
                if t < self.at[lo]:
                    continue                    # this segment hasn't started by t
                i = max(lo, bisect.bisect_right(self.at, t, lo, hi) - 1)
                n = self.an[i] + (t - self.at[i]) / self._slope(i, lo, hi)
                if j + 1 < len(self.seg) and n >= self.seg[j + 1]:
                    return float(self.seg[j + 1]), True
                return n, False
            return float(self.seg[0]), True

    def next_start(self, x: float):
        """(n_s, play_at(n_s)) of the first segment that starts after sample x, or None."""
        with self.lock:
            k = bisect.bisect_right(self.seg, x)
            if k >= len(self.seg):
                return None
            n_s = self.seg[k]
            return n_s, self.at[bisect.bisect_left(self.an, n_s)]

    def end(self):
        """One past the newest sample that has arrived (None before any)."""
        return self.content.range()[1]

    def close(self, log=print) -> bool:
        """Free the C content, unless an engine that plays it is still in a tick (it would read freed
        memory and crash the app): then it is kept, a small leak (I3)."""
        if any(e.running for e in self.readers):
            log("audio: an output engine is still running; keeping its audio")
            return False
        self.content.close()
        return True


# ----------------------------------------------------------------------------- output engine
class OutputEngine:
    """The producer: keeps the C output ring filled `ahead` seconds past where the device reads.

    For each ring index w it writes the content due when w reaches the ear: index -> host time
    (DeviceClock) -> the host's clock (to_server), less this Mac's offset -> content index (the
    StreamBuffer's stamps). The speed is fed forward (device rate x clock drift x the host's glide),
    so feedback only trims what's left, smallest first:
      |e| < 3 ms    speed nudged by at most 0.05 % (inaudible)
      |e| < 15 ms   at most 0.3 %
      |e| > 15 ms   a 5 ms crossfade straight to the right place (a splice)
    Starting (and after an added silence or a stall) it is silent until the exact frame the content
    is due, then fades in. A delay change switches at the stamped sample: an increase fades out
    there, leaves the added silence and fades in exactly on the new line; a decrease crossfades
    when the new line's sample is due.

    Python can stall for much longer than an IO buffer without a sound: the ring holds `ahead`
    (0.15-1 s, grown after stalls). A longer stall is one counted underrun, then a silent,
    frame-exact re-entry."""

    TICK, BLOCK, SLACK = 0.010, 960, 0.040
    AHEAD, AHEAD_MIN, AHEAD_MAX = 0.30, 0.15, 1.00
    SPLICE, FAST_FROM, SLOW, FAST, KP = 0.015, 0.003, 5e-4, 3e-3, 0.5
    XFADE = 240
    # Samples left unplayed at the arrival edge: the resampler reads 17 ahead, and a delay change
    # announced by the next block to arrive still finds room for its 5 ms fade-out.
    LOOK = 17 + XFADE
    ROOM = 4096             # never write within this of a full ring
    STALL_FLOOR = 0.050

    def __init__(self, ring, sb: StreamBuffer, to_server=lambda t: t, server_rate=lambda: 1.0,
                 offset_s=lambda: 0.0, volume=lambda: 1.0, channel=lambda: "both", clock_offset: float = 0.0,
                 lead_guess: float | None = None, now=time.monotonic):
        self.ring, self.sb = ring, sb
        sb.readers[:] = [e for e in sb.readers if e.running] + [self]     # (stopped ones read nothing)
        self.to_server, self.server_rate, self.offset_s = to_server, server_rate, offset_s
        self.volume, self.channel, self.now = volume, channel, now
        self.rate = ring.rate
        self.clock = DeviceClock(ring.rate, offset=clock_offset, lead_guess=lead_guess)
        self.ahead = self.AHEAD
        self.buf = np.zeros((self.BLOCK, sb.channels), np.float32)
        self.cursor = None              # fractional content index of the next frame written
        self._end = None                # one past the newest sample arrived, as of this tick
        self.silent = True
        self.splices = self.reseeks = self.steps = 0
        self.gaps = self.gap_frames = 0     # re-seeks the C side can't see as underruns (see _fill)
        # Every time the ring ran dry, was it because this thread (or the audio's arrival) was held up by
        # Python for longer than the audio it had left in the ring? That is the one thing 2.0 can't ride
        # out. If it came back in time and the ring ran dry anyway, that's a bug: `unexplained`.
        self.unexplained = 0
        self._lead_left = None              # seconds of audio in the ring after the last fill, counted
        self._fill_t = None                 # from where the device was when this was read (its time)
        self._state_t = None
        self._gap = None                # (C's underrun count, frames) of a gap until the next tick checks it
        self.err = 0.0                  # the last timing error measured (s, + = content late)
        self.sync_since = None          # since when the timing has held in sync (see _track_sync)
        self._jumps = 0
        self.max_frames = 0
        self.vol = None                 # gain at the next frame written
        self._vol_to = 1.0
        self.mode = None                # channel matrix in use, and the one it is fading from
        self._mix_from, self._mix_a = None, 1.0
        self._written: deque = deque(maxlen=128)    # (w, content x, rho, frames) of blocks played straight
        self._lin = None
        self.counts = dict.fromkeys(("callbacks", "skips", "skipped_frames", "underrun_events", "underrun_frames"), 0)
        self.last_cb = None             # PortAudio time of the newest callback
        self.stalls: deque = deque(maxlen=10)       # when callbacks stopped for >= 0.3 s
        self.headroom = None
        self.spare_1s = self.headroom_min = None
        self.history: deque = deque(maxlen=60)      # per second, for the watchdog
        self._w30: deque = deque(maxlen=30)
        self._spare30: deque = deque(maxlen=30)
        self._deficit5: deque = deque(maxlen=5)
        self._sec_t = self._tick_t = self._report_t = None
        self._h_min, self._w_max = math.inf, 0.0
        self._sec0 = dict(self.counts)
        self._sec_cb_ns = self._seen_underruns = self._c_underruns = 0
        self._quiet_since = self._last_shrink = None
        self.last_underrun = None
        self._report_reset()
        self._stop = threading.Event()
        self.thread = None

    # -------------------------------------------------------------------------- thread
    def start(self):
        self.thread = threading.Thread(target=self._run, name="output-engine", daemon=True)
        self.thread.start()
        return self

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def stop(self) -> bool:
        """Whether the thread stopped: until it has, the ring and the content must stay (I3)."""
        self._stop.set()
        if self.thread:
            self.thread.join(STOP_WAIT)
        return not self.running

    def _run(self):
        sacore.mac_call("qos_interactive")
        failed = False
        while not self._stop.wait(self.TICK):
            try:
                self.tick()
            except Exception:       # noqa: BLE001 - keep playing whatever happens
                if not failed:
                    failed = True
                    print("output engine error:\n" + traceback.format_exc(), file=sys.stderr)
                time.sleep(0.05)

    # ---------------------------------------------------------------------------- tick
    def tick(self):
        now = self.now()
        loop = now - self._tick_t if self._tick_t is not None else self.TICK    # incl. the last render
        self._tick_t = now
        if self._sec_t is None:
            self._sec_t = self._report_t = self._quiet_since = self._last_shrink = now
        recs = self.ring.new_records()
        if len(recs):
            self._callbacks(recs)
        st = self.ring.state()
        self._state_t = self.now()              # when r (where the device is) was read
        self._count(st, now)
        if self.clock.ready:
            # The arrival edge, read once a tick: each call into C lets go of the GIL, and under load
            # getting it back can take tens of ms. (What arrives meanwhile waits for the next tick.)
            self._end = self.sb.end()
            self._fill(st["r"], st["w"])
            self.headroom = self._headroom(st["r"])
        self._track_sync(now)
        self._windows(now, loop)

    def _track_sync(self, now: float):
        """In sync once |err| < 3 ms; out again beyond 10 ms or at any jump (splice, re-seek, underrun).
        The gap between keeps the label steady through a few ms of wobble, which nobody hears."""
        jumps = self.splices + self.reseeks + self.counts["underrun_events"]
        if jumps != self._jumps or self.cursor is None or abs(self.err) > 0.010:
            self.sync_since = None
        elif self.sync_since is None and not self.silent and abs(self.err) < 0.003:
            self.sync_since = now
        self._jumps = jumps

    def _callbacks(self, recs):
        self.clock.feed(recs)
        self.max_frames = max(self.max_frames, int(recs["frames"].max()))
        cur = recs["current_time"]
        gaps = np.diff(cur, prepend=cur[0] if self.last_cb is None else self.last_cb)
        if gaps.max() >= 0.3:                   # macOS stopped calling for a while
            self.stalls.append(float(cur[int(np.argmax(gaps))]))
        self.last_cb = float(cur[-1])

    def _count(self, st, now):
        if self._gap is not None:
            if st["underrun_events"] > self._gap[0]:    # the device got to the end of the old run before
                self.gaps -= 1                          # the new one was written: the C side counted it
                self.gap_frames -= self._gap[1]
            self._gap = None
        self._c_underruns = st["underrun_events"]
        for k in self.counts:
            self.counts[k] = st[k]
        self.counts["underrun_events"] += self.gaps
        self.counts["underrun_frames"] += self.gap_frames
        if self.counts["underrun_events"] > self._seen_underruns:     # the ring ran dry: keep more in it
            self.ahead = min(self.AHEAD_MAX, 1.5 * self.ahead)
            self.last_underrun = self._quiet_since = now
        self._seen_underruns = self.counts["underrun_events"]
        r = self._rep
        if st["min_lead_frames"] != sacore.NO_LEAD:
            r["lead_min"] = min(r["lead_min"], st["min_lead_frames"])
        r["flags"] |= st["flag_or"]
        r["cb_max_ns"] = max(r["cb_max_ns"], st["max_cb_ns"])
        self._sec_cb_ns = max(self._sec_cb_ns, st["max_cb_ns"])

    def _fill(self, r: int, w: int):
        guard = self.max_frames + 64            # what the device may be reading right now
        edge = w                                # where the audio in the ring ends now
        gap = False
        if w < r + guard:                       # first time, or the ring ran dry (Python stalled for
            if self.cursor is not None and not self.silent:        # longer than ahead, or the audio
                self.reseeks += 1                                   # came too late)
                if self._fill_t is not None and self._lead_left is not None and \
                        self.now() - self._fill_t < self._lead_left - guard / self.rate - 0.010:
                    self.unexplained += 1           # it was back in time, and the ring ran dry anyway
                # Some audio was left, but less than the device may take next: the new run starts past
                # it, and the silence between is audio missing like any underrun - which the C side
                # counts only when the ring runs out (edge < r), not for a gap between runs.
                gap = edge >= r
            w = r + guard + int(self.SLACK * self.rate)
            self.silent = True                  # start again exactly on time, with a fade-in
            self._written.clear()
        else:
            w = self._rewind(r + guard, w)
        end = min(r + int(self.ahead * self.rate), r + self.ring.cap - self.ROOM - self.BLOCK)
        while w < end:
            done, waiting = self._render(w, self.BLOCK)
            if done:
                if self.ring.write(w, self._mix(done), guard) < 0:
                    break                       # the device got there first: the next tick re-seeks
                if gap:                         # (counted once the new run is there to leave a gap)
                    gap = False
                    self.gaps += 1
                    self.gap_frames += w - edge
                    self._gap = (self._c_underruns, w - edge)
                if self._lin:
                    self._written.append((w, *self._lin, done))
                w += done
            if waiting:
                break
        if w < edge:            # rewound, and the new line has only arrived this far: the rest of the
            self.ring.cut(w, guard)     # ring is the old line (the new one is written as it arrives)
        self._lead_left, self._fill_t = (w - r) / self.rate, self._state_t

    def _change(self, x: float):
        """The next delay change after content x: (n_s, due on the new line, due on the old line)."""
        nxt = self.sb.next_start(x)
        if nxt is None:
            return None
        n_s, due_s = nxt
        return n_s, due_s, self.sb.play_at(n_s - 1) + self.sb.slope(n_s - 1)

    def _rewind(self, first: int, w: int) -> int:
        """A delay decrease is only known once its block arrives, and the ring may already hold the
        old line past the moment it is due. Go back and write it again from that frame (if the device
        hasn't got there), so every Mac switches at the same sample."""
        if self.silent or self.cursor is None or not self._written:
            return w
        ch = self._change(self.cursor)
        if ch is None or ch[1] >= ch[2] - 1e-4:
            return w
        T = self.to_server(self.clock.time_of(w))
        if T is None:
            return w
        late = (T - self.offset_s()) - ch[1]
        pos = w - math.ceil(late / (self.clock.per * self.server_rate()))
        if late <= 0 or pos < first:
            return w
        for w0, x0, rho, k in reversed(self._written):
            if w0 <= pos < w0 + k:
                self.cursor = x0 + (pos - w0) * rho
                return pos
        return w

    # --------------------------------------------------------------------------- render
    def _avail(self, pos: float, rho: float, n: int) -> int:
        """How many of n frames from content position pos (step rho) have arrived."""
        end = self._end
        if end is None:
            return 0
        return max(0, min(n, math.floor((end - self.LOOK - pos) / rho) + 1))

    def _gain(self, j: int) -> float:
        """Volume at frame j of this block: changes glide at most full scale per BLOCK frames."""
        d = self._vol_to - self.vol
        return self.vol + max(-j / self.BLOCK, min(j / self.BLOCK, d))

    def _render(self, w: int, n: int):
        """Audio for ring positions w... into self.buf. Returns (frames ready, waiting): waiting means
        what comes next hasn't arrived (or the clocks aren't known yet)."""
        y = self.buf[:n]
        y.fill(0.0)
        self._lin = None
        sb = self.sb
        T = self.to_server(self.clock.time_of(w))
        if T is None:
            return 0, True
        T -= self.offset_s()
        per = self.clock.per * self.server_rate()       # host seconds per device frame
        self._vol_to = float(self.volume())
        if self.vol is None:
            self.vol = self._vol_to
        if self.cursor is None:
            x, _ = sb.index_at(T)
            if x is None:
                return 0, True
            self.cursor, self.silent = x, True
        done, waiting = self._enter(T, per, y, n) if self.silent else self._play(T, per, y, n)
        self.vol = self._gain(done)
        return done, waiting

    def _enter(self, T, per, y, n):
        """Silence until the exact frame the content is due, then a 5 ms fade-in."""
        sb, x = self.sb, self.cursor
        e = T - sb.play_at(x)
        if e > self.SPLICE:                     # what's due now is further on: start there
            x, gap = sb.index_at(T)
            self.cursor = x
            if gap:                             # inside the silence a delay increase added
                return n, False
            e = T - sb.play_at(x)
        m = max(0, round(-e / per))            # silent frames before x is due
        if m >= n or (m and n - m < self.XFADE):
            return m if m < n else n, False     # (the next block starts with the whole fade-in)
        slope = sb.slope(x)
        start = x + (e + m * per) / slope       # the content at frame m, to a fraction of a sample
        rho = per / slope
        ch = self._change(x)
        if ch is not None:                      # a delay change in this block: the old line stops where
            n_s, due_s, old_due = ch            # an increase's fade-out starts, or a decrease is due
            stop = (m + math.ceil((n_s - start) / rho) - self.XFADE if due_s > old_due + 1e-4
                    else round((due_s - T) / per) if due_s < old_due - 1e-4 else n)
            if stop - m < self.XFADE:           # no room to fade in before that: enter on the new line
                self.cursor = float(n_s)
                self.steps += 1
                return m, False
            n = min(n, stop)                    # (_play takes it from there)
        k = self._avail(start, rho, n - m)
        if k < min(n - m, self.XFADE):
            return m, True                      # the silence is ready; the music isn't here yet
        f = min(k, self.XFADE)
        c = sb.content
        c.resample(start, rho, f, y[m:], 0.0, self._gain(m + f - 1))
        if k > f:
            c.resample(start + f * rho, rho, k - f, y[m + f:], self._gain(m + f), self._gain(m + k - 1))
        self.cursor, self.silent = start + k * rho, False
        return m + k, k < n - m

    def _play(self, T, per, y, n):
        sb, x, c = self.sb, self.cursor, self.sb.content
        e = T - sb.play_at(x)                   # > 0: this content should already be playing
        self.err = e
        rho0 = per / sb.slope(x)                # content samples per device frame
        if abs(e) > self.SPLICE:
            return self._splice(T, per, x, rho0, y, n)
        self._rep["err_sum"] += abs(e)
        self._rep["err_n"] += 1
        self._rep["err_max"] = max(self._rep["err_max"], abs(e))
        lim = self.SLOW if abs(e) < self.FAST_FROM else self.FAST
        rho = rho0 * (1 + max(-lim, min(lim, self.KP * e)))
        ch = self._change(x)
        if ch is not None:                      # a delay change ahead
            n_s, due_s, old_due = ch
            if due_s > old_due + 1e-4:          # an increase: the old content ends at n_s
                jj = max(0, math.ceil((n_s - x) / rho))
                if jj <= self.XFADE:
                    return self._fade_out(x, rho, y, jj, n_s)
                n = min(n, jj - self.XFADE)     # play on up to where its fade-out starts
            elif due_s < old_due - 1e-4:        # a decrease: n_s is due before the old line gets there
                m = round((due_s - T) / per)
                if m <= 0:
                    return self._cross_into(x, rho, n_s + (T - due_s) / sb.slope(n_s), per, y, n, step=True)
                n = min(n, m)                   # play on until n_s is due
        k = self._avail(x, rho, n)
        if k:
            c.resample(x, rho, k, y, self._gain(0), self._gain(k - 1))
            self.cursor = x + k * rho
            self._lin = (x, rho)
        return k, k < n

    def _fade_out(self, x, rho, y, jj, n_s):
        """The last jj frames before a delay increase at n_s, fading out; then silence until n_s is
        due on the new line (the silent start lines that up to the frame)."""
        if self._avail(x, rho, jj) < jj:
            return 0, True
        if jj:
            self.sb.content.resample(x, rho, jj, y, self._gain(0), 0.0)
        self.cursor, self.silent = float(n_s), True
        self.steps += 1
        return jj, False

    def _cross_into(self, x, rho, target, per, y, n, step=False):
        """5 ms crossfade from content x to content target (a decrease's new line, or a splice)."""
        c = self.sb.content
        rho_t = per / self.sb.slope(target)
        f = min(n, self.XFADE)
        k = self._avail(target, rho_t, n)
        if k < f or self._avail(x, rho, f) < f:
            return 0, True
        c.resample(x, rho, f, y, self._gain(0), 0.0)                       # the old one fades out...
        c.resample(target, rho_t, f, y, 0.0, self._gain(f - 1), add=True)  # ...the right one fades in
        if k > f:
            c.resample(target + f * rho_t, rho_t, k - f, y[f:], self._gain(f), self._gain(k - 1))
        self.cursor = target + k * rho_t
        if step:
            self.steps += 1
        else:
            self.splices += 1
        return k, k < n

    def _splice(self, T, per, x, rho0, y, n):
        target, gap = self.sb.index_at(T)
        if not gap:
            return self._cross_into(x, rho0, target, per, y, n)
        f = min(n, self.XFADE)                  # inside an added silence: fade out into it
        if self._avail(x, rho0, f) < f:
            return 0, True
        self.sb.content.resample(x, rho0, f, y, self._gain(0), 0.0)
        self.cursor, self.silent = target, True
        self.splices += 1
        return f, False

    def _matrix(self, mode: str) -> np.ndarray:
        """Content channels -> device channels: both / left / right, and mono devices get the mix."""
        cin, cout = self.sb.channels, self.ring.channels
        m = np.zeros((cin, cout), np.float32)
        if cin == 1 or mode == "left":
            m[0, :] = 1
        elif mode == "right":
            m[1, :] = 1
        elif cout == 1:
            m[:, 0] = 1 / cin
        else:
            m[range(min(cin, cout)), range(min(cin, cout))] = 1
        return m

    def _mix(self, done: int) -> np.ndarray:
        y = self.buf[:done]
        mode = self.channel()
        if mode != self.mode:                   # glide to the new mix over a block, no step
            if self.mode is not None:
                self._mix_from, self._mix_a = self._matrix(self.mode), 0.0
            self.mode, self._m = mode, self._matrix(mode)
        if self._mix_from is None:
            if self.sb.channels == self.ring.channels and mode == "both":
                return y
            return y @ self._m
        a = np.minimum(1.0, self._mix_a + np.arange(1, done + 1, dtype=np.float32) / self.BLOCK)[:, None]
        self._mix_a = float(a[-1, 0])
        out = (y @ self._mix_from) * (1 - a) + (y @ self._m) * a
        if self._mix_a >= 1.0:
            self._mix_from = None
        return out.astype(np.float32)

    # ----------------------------------------------------------------------- bookkeeping
    def _headroom(self, r: int):
        """How long the audio already here lasts beyond the frame the device reads next (s, on true
        time): what the delay has to keep above the producer's stall allowance."""
        end = self._end
        if end is None or self.cursor is None:
            return None
        T = self.to_server(self.clock.time_of(r))
        return None if T is None else self.sb.play_at(end - self.LOOK) - (T - self.offset_s())

    def _windows(self, now: float, loop: float):
        """Per-second minima and maxima, and `ahead` adapted to the worst loop period in 30 s."""
        if self.headroom is not None:
            # Right after this Mac's Python stalled, what arrived meanwhile isn't in yet (the network or
            # capture thread stalled too): that dip is the stall itself, which the stall allowance below
            # already makes room for. Counting it here as well would count it twice.
            self._h_min = min(self._h_min, self.headroom + (loop - self.TICK if loop > 3 * self.TICK else 0.0))
        self._w_max = max(self._w_max, loop)
        self._rep["loop_max"] = max(self._rep["loop_max"], loop)
        if now - self._sec_t >= 1.0:
            self._sec_t = now if now - self._sec_t > 2.0 else self._sec_t + 1.0
            self._w30.append(self._w_max)
            if math.isfinite(self._h_min):
                self.headroom_min = self._h_min
                self.spare_1s = self._h_min - self.stall_allowance()
                self._spare30.append(self.spare_1s)
                self._deficit5.append(max(0.0, -self._h_min))
            self.history.append({"t": now, "cb_max_us": self._sec_cb_ns / 1000,
                                 **{k: v - self._sec0[k] for k, v in self.counts.items()}})
            self._sec0, self._sec_cb_ns = dict(self.counts), 0
            self._h_min, self._w_max = math.inf, 0.0
        w30 = max([self._w_max, *self._w30])
        # Just after the device takes a buffer, a buffer and the guard must still be left (else every
        # callback is a re-seek: 4096-frame buffers would lose half their audio at 0.15 s), through
        # the worst stall in 30 s.
        target = max(self.AHEAD_MIN, 1.5 * w30 + (2 * self.max_frames + 64) / self.rate)
        self.ahead = min(self.AHEAD_MAX, max(self.ahead, target))
        if now - self._quiet_since >= 60 and now - self._last_shrink >= 10:
            self.ahead = max(target, 0.9 * self.ahead)
            self._last_shrink = now

    def stall_allowance(self) -> float:
        """How long a Python stall the ring must ride out: 1.5 x the worst loop period in 30 s."""
        return max(self.STALL_FLOOR, 1.5 * max([self._w_max, *self._w30]))

    @property
    def spare_30(self):
        return min(self._spare30) if self._spare30 else None

    def _report_reset(self):
        self._rep = {"lead_min": sacore.NO_LEAD, "flags": 0, "cb_max_ns": 0, "err_sum": 0.0, "err_n": 0,
                     "err_max": 0.0, "loop_max": 0.0, "callbacks": self.counts["callbacks"]}

    def second(self) -> dict:
        """This output's numbers for the last second (diagnostics 11.2); starts the next second."""
        now = self.now()
        rep, c, cs = self._rep, self.clock, self.counts
        dt = max(1e-9, now - (self._report_t or now))
        ms = lambda s: None if s is None else round(s * 1000, 1)        # noqa: E731
        lead = c.lead
        out = {"io": [c.frames_min, c.frames_max],
               "cb_per_s": round((cs["callbacks"] - rep["callbacks"]) / dt, 1) if self._report_t != now else None,
               "cb_max_us": round(rep["cb_max_ns"] / 1000, 1),
               "skips": cs["skips"], "skipped_ms": round(cs["skipped_frames"] / self.rate * 1000, 1),
               "pa_flags": rep["flags"], "underruns": cs["underrun_events"], "unexplained": self.unexplained,
               "underrun_ms": round(cs["underrun_frames"] / self.rate * 1000, 1),
               "ring_lead_min_ms": None if rep["lead_min"] == sacore.NO_LEAD else ms(rep["lead_min"] / self.rate),
               "ahead_ms": ms(self.ahead), "headroom_min_ms": ms(self.headroom_min),
               "spare_1s_ms": ms(self.spare_1s), "spare_30_ms": ms(self.spare_30),
               "deficit_ms": ms(max(self._deficit5, default=0.0)),
               "err_ms": round(rep["err_sum"] / rep["err_n"] * 1000, 3) if rep["err_n"] else None,
               "err_max_ms": round(rep["err_max"] * 1000, 3),
               "splices": self.splices, "reseeks": self.reseeks, "steps": self.steps,
               "dll_ppm": round(c.rate_ppm, 1), "dll_jitter_ms": c.stats()["jitter_ms"],
               "dll_reanchors": c.reanchors, "lossy": c.lossy, "loss_pct": round(c.loss_pct, 2),
               "producer_max_ms": ms(rep["loop_max"]),
               "lead_ms": None if lead is None else ms(lead + c.offset)}
        self._report_t = now
        self._report_reset()
        return out


# ---------------------------------------------------------------------------- capture engine
class CaptureEngine:
    """Drains the capture ring (BlackHole) in 10 ms blocks, each stamped with when it was captured.

    The C input callback puts every frame at its own device index, and frames macOS skipped as zeros
    at theirs, so an index is a time: a block's n is its device index plus a base, and its capture
    time comes from the input's own DLL (ADC stamps). If this thread ever falls a whole ring (5.4 s)
    behind, it jumps to 50 ms behind the newest frame: n (and so time) stays continuous, and the
    players play what was lost as silence. A reopened device carries on with n where real time says
    (`after`: the next n and its capture time from the stream before)."""

    PERIOD, BLOCK = 0.005, 480

    def __init__(self, ring, on_block, stream_latency: float = 0.0, after=None, lead_guess=None,
                 now=time.monotonic):
        self.ring, self.on_block, self.now, self.after = ring, on_block, now, after
        self.rate = ring.rate
        self.clock = DeviceClock(ring.rate, offset=-stream_latency, lead_guess=lead_guess)
        self.rpos = self._first = self.last_cb = None
        self.n_base = 0
        self.lapped = self.gap_frames = 0
        self.counts = {"skips": 0, "skipped_frames": 0, "overflows": 0, "callbacks": 0}
        self.reader_max = 0.0
        self._failed = False
        self._stop = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name="capture-engine", daemon=True)
        self.thread.start()
        return self

    def stop(self) -> bool:
        """Whether the thread stopped: until it has, the ring must stay (I3)."""
        self._stop.set()
        if self.thread:
            self.thread.join(STOP_WAIT)
        return not (self.thread and self.thread.is_alive())

    def _run(self):
        sacore.mac_call("qos_interactive")
        while not self._stop.wait(self.PERIOD):
            try:
                self.tick()
            except Exception:       # noqa: BLE001 - keep capturing whatever happens
                if not self._failed:
                    self._failed = True
                    print("capture engine error:\n" + traceback.format_exc(), file=sys.stderr)
                time.sleep(0.05)

    def tick(self):
        t0 = self.now()
        recs = self.ring.new_records()
        if len(recs):
            if self._first is None:
                self._first = int(recs["pos"][0])
            self.clock.feed(recs)
            self.last_cb = float(recs["current_time"][-1])
        st = self.ring.state()
        for k in self.counts:
            self.counts[k] = st[k]
        if not self.clock.ready:
            return
        w = st["w"]
        if self.rpos is None:
            self.rpos = self._first
            self.n_base = -self.rpos
            if self.after:                      # carry on where real time says, from the last stream
                last_n, last_t = self.after
                n0 = last_n + round((self.clock.time_of(self.rpos) - last_t) * self.rate)
                self.gap_frames += n0 - last_n
                self.n_base = n0 - self.rpos
        while w - self.rpos >= self.BLOCK:
            try:
                pcm = self.ring.read(self.rpos, self.BLOCK)
            except sacore.Lapped:
                self.lapped += 1
                # 50 ms behind the newest frame now, not as of the start of this tick: this thread may
                # have stalled since (for another ring, and then that position would be lost too)
                w = self.ring.state()["w"]
                self.rpos = w - 5 * self.BLOCK
                continue
            if pcm is None:
                break
            try:
                self.on_block(pcm, self.n_base + self.rpos, self.clock.time_of(self.rpos))
            except Exception:       # noqa: BLE001 - one bad block must not stop capture
                if not self._failed:
                    self._failed = True
                    print("capture: on_block failed:\n" + traceback.format_exc(), file=sys.stderr)
            self.rpos += self.BLOCK
        self.reader_max = max(self.reader_max, self.now() - t0)

    def continuity(self):
        """(next n, its capture time): where a reopened capture should carry on."""
        if self.rpos is None:
            return self.after
        return self.n_base + self.rpos, self.clock.time_of(self.rpos)

    def second(self) -> dict:
        c = self.clock
        out = {"cap_io": [c.frames_min, c.frames_max], "cap_skips": self.counts["skips"],
               "cap_skipped_ms": round(self.counts["skipped_frames"] / self.rate * 1000, 1),
               "cap_overflows": self.lapped, "cap_gap_frames": self.gap_frames,
               "cap_reader_max_ms": round(self.reader_max * 1000, 1), "cap_dll_ppm": round(c.rate_ppm, 1)}
        self.reader_max = 0.0
        return out


# ----------------------------------------------------------------------- opening the streams
def _pa_facts(sd, index: int, kind: str, stream) -> dict:
    """PortAudio's own figures for an open stream. Read once, at open: a call into PortAudio from another
    thread (the app polls info()) while a session restarts PortAudio would read memory it just freed."""
    try:
        info = sd.query_devices(index)
        return {"portaudio": sd.get_portaudio_version()[1], "sounddevice": sd.__version__,
                "default_low_ms": round(info[f"default_low_{kind}_latency"] * 1000, 1),
                "default_high_ms": round(info[f"default_high_{kind}_latency"] * 1000, 1),
                "stream_latency_ms": round(stream.latency * 1000, 1)}
    except Exception:       # noqa: BLE001 - diagnostics only (and the tests' fakes)
        return {}


class _Stream:
    """A PortAudio stream on a libsacore ring: PortAudio calls the C callback directly (its address
    and the ring as userdata), so no Python - and no GIL - is ever on CoreAudio's IO thread."""

    def __init__(self, sd, kind: str, index: int, rate: float, channels: int, blocksize: int, latency: float):
        self.ring = sacore.Ring(kind, channels, rate)
        try:
            self.stream = sd._StreamBase(kind=kind, samplerate=rate, blocksize=blocksize, device=index,
                                         channels=channels, dtype="float32", latency=latency,
                                         callback=self.ring.callback, userdata=sacore.userdata(sd, self.ring.h),
                                         wrap_callback=None)
        except Exception:
            self.ring.close()
            raise

    def close(self, log, free: bool = True) -> bool:
        """Stop PortAudio first; free the ring only once it can't call into it any more, and only if
        `free` (its engine thread has stopped). Returns whether the stream closed."""
        try:
            self.stream.abort()     # (an error here may just mean it had stopped already)
            # Pa_CloseStream aborts the stream again if it isn't stopped, and closes it only if that
            # worked: sounddevice ignores its errors unless asked, and an error means the IO thread may
            # still be calling into the ring.
            self.stream.close(ignore_errors=False)
        except Exception as e:      # noqa: BLE001 - leak the ring rather than free it under PortAudio
            log(f"audio: closing the stream failed ({e!r}); keeping its buffer")
            return False
        if free:
            self.ring.close()
        return True


class AudioOutput:
    """One output device: facts, plan, ring, stream, listeners, then the producer.

    Opens in that order (and closes in reverse): the device's facts (sacore.mac, if present), the
    IO plan, a ring, the stream with the C callback, which CoreAudio device it really is, listeners for
    overloads / latency / rate / alive, start, then the OutputEngine, attached to the player. The
    player is what plays here: .sb (a StreamBuffer), .to_server, .server_rate, attach() / detach()."""

    def __init__(self, sd, index: int, rate: float, player, settings, prefs_dev=None, log=print,
                 reason: str = "start", level: int | None = None, now=time.monotonic, threads: bool = True):
        self.sd, self.index, self.player, self.log, self.now, self.reason = sd, index, player, log, now, reason
        info = sd.query_devices(index)
        self.name = info["name"]
        self.channels = min(2, info["max_output_channels"])
        self.rate = rate
        self.facts = sacore.mac_call("device_info", self.name, True)
        saved = level is None
        if saved:
            level = (prefs_dev or {}).get("io_level", 0)
        self.plan = plan_output(sd, index, rate, self.facts, level)
        if self.plan.level:
            self.plan.reason = "saved for this device" if saved else reason
        nominal = (self.facts or {}).get("nominal_rate") or rate
        self.call_profile = self.channels == 1 or nominal <= 24000     # a headset in a call: mono, low rate
        s = _Stream(sd, "output", index, rate, self.channels, self.plan.blocksize, self.plan.latency)
        self.ring, self.stream, self._s = s.ring, s.stream, s
        self.listeners, self.engine = {}, None
        try:            # (AirPods can vanish between open and start: then nothing may be left behind)
            self.dev_id = sacore.mac_call("stream_device", sd, self.stream, True)
            if self.dev_id:
                self.facts = sacore.mac_call("device_info_id", self.dev_id, True) or self.facts
            self.listeners = self._listen()
            self._seen = {k: c.value for k, c in self.listeners.items()}
            strlat = self.facts["stream_latency"] / nominal if self.facts else 0.0
            self.engine = OutputEngine(self.ring, player.sb, to_server=player.to_server,
                                       server_rate=player.server_rate, offset_s=lambda: settings.offset_ms / 1000,
                                       volume=lambda: settings.volume, channel=lambda: settings.channel,
                                       clock_offset=strlat, lead_guess=getattr(self.stream, "latency", None), now=now)
            self.warning = None
            self._frames_checked = False
            self.pa = _pa_facts(sd, index, "output", self.stream)
            self.stream.start()
            self.opened = now()
            if threads:
                self.engine.start()
        except BaseException:
            self._release()
            raise
        player.attach(self.engine)
        if threads:                         # the first callbacks, to see what PortAudio really gave
            while self.engine.clock.records < 3 and now() - self.opened < 1.0:
                time.sleep(0.01)
        log(f"audio: {self.name} plays in {self.engine.clock.frames_max or '?'}-frame buffers "
            f"(plan {self.plan.blocksize or 'auto'}, level {self.plan.level}, {reason})")

    def _listen(self) -> dict:
        sel = getattr(sacore.mac, "SEL", None)
        if not (self.dev_id and sel):
            return {}
        out = {}
        for key, what, scope in (("overload_global", "overload", "global"),
                                 ("overload_wildcard", "overload", "wildcard"), ("latency", "latency", "output"),
                                 ("rate", "nominal_rate", "global"), ("alive", "alive", "global")):
            c = sacore.mac_call("listen", self.dev_id, sel[what], sel[scope])
            if c:
                out[key] = c
        return out

    @property
    def overloads(self) -> int:
        """macOS's own overload notifications. The wildcard listener also hears the global ones, so
        the larger count is the number of events, not the sum."""
        return max((self.listeners[k].value for k in ("overload_global", "overload_wildcard") if k in self.listeners),
                   default=0)

    @property
    def latency_changes(self) -> int:
        """How often macOS said the device's latency changed. PortAudio moves its timestamps with it, so a
        skip (or a new clock line) at that moment is the new latency, not audio macOS dropped."""
        c = self.listeners.get("latency")
        return c.value if c else 0

    def _changed(self, key: str) -> bool:
        c = self.listeners.get(key)
        if c is None or c.value == self._seen[key]:
            return False
        self._seen[key] = c.value
        return True

    def _reopen(self, level: int, reason: str, persist: bool = False) -> dict:
        return {"action": "reopen", "level": level, "from": self.plan.level, "reason": reason, "persist": persist}

    def watchdog(self):
        """Called once a second by the session: what should happen to this output, or None.
        {"action": "reopen", "level", "reason", "persist"}: open it again with that plan level (persist:
        remember the level for this device); {"action": "follow"}: the device is gone, play on the
        current default output; {"action": "warn", "warning"}: tell the user."""
        now, e = self.now(), self.engine
        hist, stalls = tuple(e.history), tuple(e.stalls)    # the engine thread appends to these
        if self._changed("alive") and (sacore.mac_call("device_info_id", self.dev_id, True) or {}).get("alive", 1) == 0:
            return {"action": "follow", "reason": "device_gone"}
        last = e.last_cb if e.last_cb is not None else self.opened
        if getattr(self.stream, "active", True) and now - last >= 1.0:
            return {"action": "follow", "reason": "no_callbacks"}
        if len([t for t in stalls if now - t <= 10]) >= 2 and self.plan.level < 2:
            # e.g. macOS 27 Bluetooth wanting bigger slices than PortAudio allows: let it choose
            return self._reopen(2, "callbacks_stall", persist=True)
        if self._changed("rate"):
            return self._reopen(self.plan.level, "rate_changed")
        if not self._frames_checked and now - self.opened >= 1.0:
            self._frames_checked = True
            if self.plan.level == 0 and e.clock.frames_max is not None and e.clock.frames_max < MIN_OK_FRAMES:
                return self._reopen(1, "small_buffer")
        last30 = [h for h in hist if now - h["t"] <= 30]
        skips = sum(h["skips"] for h in last30)
        if self.warning and not any(h["skips"] for h in hist if now - h["t"] <= 60):
            self.warning = None                 # a quiet minute
        if skips >= 3 and not any(h["underrun_events"] for h in last30):
            if self.plan.level == 0 and max(h["cb_max_us"] for h in last30) < 1000:
                return self._reopen(1, "macos_skipping", persist=True)     # macOS overloads by itself
            if self.warning != "output_dropping":
                self.warning = "output_dropping"
                return {"action": "warn", "warning": "output_dropping",
                        "count": sum(h["skips"] for h in hist if now - h["t"] <= 60)}
        return None

    def second(self) -> dict:
        out = self.engine.second()
        out.update(out=self.name, uid=(self.facts or {}).get("uid"), overloads=self.overloads,
                   latency_changes=self.latency_changes)
        return out

    def info(self) -> dict:
        """What was opened and what PortAudio really did (diagnostics 11.1)."""
        f, c, p = self.facts, self.engine.clock, self.plan
        # PortAudio's DAC time - its "now" should be the IO buffer + safety offset + device latency
        props = ((p.blocksize or c.frames_max or 0) + f["safety"] + f["latency"]) / self.rate * 1000 if f else None
        lead = c.first_lead if c.first_lead is not None else c.lead
        probe = lead * 1000 if lead is not None else None
        dev = None
        if f:
            dev = {k: f[k] for k in ("name", "uid", "transport", "alive", "nominal_rate", "actual_rate",
                                     "out_channels", "buffer_frames", "buffer_min", "buffer_max")}
            dev.update(latency_frames=f["latency"], safety_frames=f["safety"],
                       stream_latency_frames=f["stream_latency"])
        return {"engine": ENGINE, "sacore": sacore.LIBRARY, "sacore_version": hex(sacore.VERSION or 0),
                "device": dev, "pa": self.pa,
                "plan": {"blocksize": p.blocksize, "latency_s": p.latency, "level": p.level, "bt_class": p.bt_class,
                         "fixed_ms": round(p.fixed_s * 1000, 1) if p.fixed_s is not None else None,
                         "source": p.source, "reason": p.reason},
                "observed": {"frames_min": c.frames_min, "frames_max": c.frames_max,
                             "callbacks_per_s": round(c.records / max(1e-9, self.now() - self.opened), 1),
                             "lead_ms": c.stats()["lead_ms"],
                             "probe_d_ms": round(probe, 1) if probe is not None else None,
                             "props_d_ms": round(props, 1) if props is not None else None,
                             "d_mismatch_ms": round(probe - props, 1) if None not in (probe, props) else None},
                "src_active": bool(f and f["nominal_rate"] and f["nominal_rate"] != self.rate),
                "call_profile": self.call_profile, "stream_rate": self.rate, "opened_reason": self.reason,
                "warning": self.warning, "skips_60s": sum(h["skips"] for h in tuple(self.engine.history))}

    def _release(self) -> bool:
        stopped = self.engine is None or self.engine.stop()
        if not stopped:
            self.log("audio: the output engine didn't stop in time; keeping its buffers")
        closed = self._s.close(self.log, free=stopped)
        for c in self.listeners.values():
            try:
                c.close()
            except Exception as e:      # noqa: BLE001 - the macOS helpers are optional
                self.log(f"audio: removing a listener failed: {e!r}")
        return stopped and closed

    def close(self) -> bool:
        """Returns whether everything was freed (else some memory was kept: see I3)."""
        self.player.detach()
        return self._release()


class AudioInput:
    """The capture device (BlackHole): a C input ring, its stream (512-frame buffers, latency 0) and
    the CaptureEngine calling on_block(pcm, n, captured_at). `after`: continuity() of the input this
    one replaces, so n carries on in real time."""

    def __init__(self, sd, index: int, rate: float, channels: int, on_block, after=None, log=print,
                 now=time.monotonic, threads: bool = True):
        self.sd, self.index, self.rate, self.log, self.now = sd, index, rate, log, now
        self.name = sd.query_devices(index)["name"]
        self.facts = sacore.mac_call("device_info", self.name, False)
        s = _Stream(sd, "input", index, rate, channels, IN_BLOCK, PA_LATENCY)
        self.ring, self.stream, self._s = s.ring, s.stream, s
        self.capture = self.closed = self._seen = None
        try:
            self.dev_id = sacore.mac_call("stream_device", sd, self.stream, False)
            if self.dev_id:
                self.facts = sacore.mac_call("device_info_id", self.dev_id, False) or self.facts
            f = self.facts
            strlat = f["stream_latency"] / f["nominal_rate"] if f and f["nominal_rate"] else 0.0
            latency = getattr(self.stream, "latency", None)
            self.capture = CaptureEngine(self.ring, on_block, strlat, after, -latency if latency else None, now)
            self.pa = _pa_facts(sd, index, "input", self.stream)
            self.stream.start()
            self.opened = now()
            if threads:
                self.capture.start()
        except BaseException:
            self.close()
            raise

    def continuity(self):
        return self.capture.continuity()

    def watchdog(self):
        """None, or {"action": "reopen"} when the device stopped calling back for a second."""
        last = self.capture.last_cb if self.capture.last_cb is not None else self.opened
        if getattr(self.stream, "active", True) and self.now() - last >= 1.0:
            return {"action": "reopen", "reason": "no_callbacks"}
        return None

    def missed(self) -> bool:
        """Whether this capture lost audio since the last call: BlackHole stopped calling back (its
        newest frame is over 0.1 s old), macOS skipped its cycles (zeros in their place), or capture fell
        a ring behind. Everyone plays that as silence, and what follows is as early as ever. Read from
        the C ring, so a stall of Python's own capture thread doesn't look like one."""
        st, c = self.ring.state(), self.capture
        seen, self._seen = self._seen, (st["skips"], c.lapped)
        late = c.clock.ready and self.now() - c.clock.time_of(st["w"]) > 0.1
        return late or (seen is not None and seen != self._seen)

    def second(self) -> dict:
        return self.capture.second()

    def info(self) -> dict:
        c = self.capture.clock
        return {"engine": ENGINE, "device": self.facts, "pa": self.pa,
                "plan": {"blocksize": IN_BLOCK, "latency_s": PA_LATENCY},
                "observed": {"frames_min": c.frames_min, "frames_max": c.frames_max, "lead_ms": c.stats()["lead_ms"]}}

    def close(self) -> bool:
        """Returns whether everything was freed (else the ring was kept: see I3). Once: a session may
        hold on to a closed capture until it can open another."""
        if self.closed is None:
            stopped = self.capture is None or self.capture.stop()
            if not stopped:
                self.log("audio: the capture engine didn't stop in time; keeping its buffer")
            self.closed = self._s.close(self.log, free=stopped) and stopped
        return self.closed


# ---------------------------------------------------------------------------------- self-test
class LocalPlayer:
    """The least a player is: a StreamBuffer on this Mac's own clock (the self-test's tone). The app's
    syncaudio.Player is the full one."""

    def __init__(self, rate: int, channels: int):
        self.sb = StreamBuffer(rate, channels)
        self.engine = None

    @staticmethod
    def to_server(t: float) -> float:
        return t

    @staticmethod
    def server_rate() -> float:
        return 1.0

    def attach(self, engine):
        self.engine = engine

    def detach(self):
        self.engine = None


def verdict(v2: dict) -> str:
    """clean: nothing skipped, overloaded, lost or run dry. macos_dropping: macOS skipped or lost audio
    while our callback took under 1 ms (not SyncAudio's timing). app_problem: the ring ran dry, or our
    callback itself was slow."""
    dropped = v2["skips"] > 0 or v2["overloads"] > 0 or v2["lossy"]
    if v2["underruns"] > 0 or (dropped and v2["cb_max_us"] >= 1000):
        return "app_problem"
    return "macos_dropping" if dropped else "clean"


class SelfTest:
    """"Test my headphones": a quiet 440 Hz tone (-30 dBFS), first the way 1.1.6 played it - a Python
    callback, latency 0.1, PortAudio's own choice of buffer: it may crackle - then through this engine,
    and a verdict (see verdict()). `phase` and `result` can be watched from another thread."""

    SECONDS = (10.0, 20.0)
    LEVEL = 10 ** (-30 / 20)

    def __init__(self, sd, index: int, rate: int = 48000, legacy: bool = True, log=print, seconds=None):
        self.sd, self.index, self.rate, self.legacy, self.log = sd, index, rate, legacy, log
        self.seconds = seconds or self.SECONDS
        self.phase = self.result = None

    def _tone(self, channels: int) -> np.ndarray:
        """Two seconds of tone. It repeats every second (440 whole cycles), so a block of up to a second
        can start anywhere in the first."""
        t = np.arange(2 * self.rate) / self.rate
        return np.repeat((self.LEVEL * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[:, None], channels, axis=1)

    def run(self) -> dict:
        res = {"output": self.sd.query_devices(self.index)["name"], "rate": self.rate}
        if self.legacy:
            self.phase = "legacy"
            res["legacy"] = self._legacy()
        self.phase = "v2"
        res["v2"] = self._v2()
        res["verdict"] = verdict(res["v2"])
        self.phase, self.result = "done", res
        return res

    def _legacy(self) -> dict:
        """The one stream the app ever opens with a Python callback: the old way, for comparison."""
        rate, n = self.rate, 1 << 16
        ch = min(2, self.sd.query_devices(self.index)["max_output_channels"])
        tone = self._tone(ch)
        frames, dac, flags = np.zeros(n, np.int64), np.zeros(n), np.zeros(n, np.int64)
        at = [0, 0]                             # callbacks so far, tone position

        def callback(outdata, k, t, status):
            i, p = at
            if i < n:
                frames[i], dac[i], flags[i] = k, t.outputBufferDacTime, getattr(status, "_flags", 0)
                at[0] = i + 1
            outdata[:] = tone[p:p + k]
            at[1] = (p + k) % rate
        stream = self.sd.OutputStream(samplerate=rate, device=self.index, channels=ch, dtype="float32",
                                      latency=0.1, callback=callback)
        stream.start()
        time.sleep(self.seconds[0])
        stream.stop()
        stream.close()
        m = at[0]
        f, d = frames[:m], dac[:m]
        start = int(np.searchsorted(np.cumsum(f), rate // 2))          # leave out the first 0.5 s
        ok = (d[start + 1:] > 0) & (d[start:-1] > 0) if m > start + 1 else np.zeros(0, bool)
        gaps = (np.diff(d[start:]) - f[start:-1] / rate)[ok]
        span = d[start:][d[start:] > 0]
        span = span[-1] - span[0] if len(span) > 1 else 0.0
        return {"frames_min": int(f.min()) if m else None, "frames_max": int(f.max()) if m else None,
                "callbacks": m, "lost_ms_per_s": round(float(gaps.sum()) / span * 1000, 2) if span > 0 else None,
                "pa_flags": int(np.bitwise_or.reduce(flags[:m])) if m else 0}

    def _v2(self) -> dict:
        player = LocalPlayer(self.rate, 2)
        settings = types.SimpleNamespace(offset_ms=0.0, volume=1.0, channel="both")
        tone = self._tone(2)
        out = AudioOutput(self.sd, self.index, self.rate, player, settings, log=self.log, reason="selftest")
        try:
            t0 = time.monotonic() + 0.4                 # the tone's first sample is due then
            end, n, cb_max, tick = time.monotonic() + self.seconds[1], 0, 0.0, time.monotonic()
            while time.monotonic() < end:
                while t0 + n / self.rate < time.monotonic() + 1.0:
                    p = n % self.rate
                    player.sb.feed(n, t0 + n / self.rate, tone[p:p + 480])
                    n += 480
                if time.monotonic() - tick >= 1.0:
                    cb_max, tick = max(cb_max, out.second()["cb_max_us"]), time.monotonic()
                time.sleep(0.05)
            st = out.second()
            c = out.engine.clock.stats()
            return {"io_frames": c["frames_max"], "io": st["io"], "plan": asdict(out.plan),
                    "skips": st["skips"], "skipped_ms": st["skipped_ms"], "overloads": st["overloads"],
                    "underruns": st["underruns"], "rate_ppm": c["rate_ppm"], "jitter_ms": c["jitter_ms"],
                    "cb_max_us": max(cb_max, st["cb_max_us"]), "lossy": c["lossy"], "loss_pct": c["loss_pct"]}
        finally:
            out.close()
            player.sb.close(self.log)


# ------------------------------------------------------------------------------ health probes
class GcTimer:
    """How long Python's garbage collector held every thread, per second (gc.callbacks)."""

    def __init__(self, now=time.perf_counter):
        self.now = now
        self._t = None
        self.max, self.gen2, self.top = 0.0, 0, []      # this second's worst; gen-2 runs; worst 5 ever

    def start(self):
        if self._cb not in gc.callbacks:
            gc.callbacks.append(self._cb)
        return self

    def stop(self):
        if self._cb in gc.callbacks:
            gc.callbacks.remove(self._cb)

    def _cb(self, phase, info):
        if phase == "start":
            self._t = self.now()
        elif self._t is not None:
            dt, self._t = self.now() - self._t, None
            self.max = max(self.max, dt)
            self.gen2 += info.get("generation") == 2
            self.top = sorted(self.top + [dt])[-5:]

    def second(self) -> dict:
        out = {"gc_max_ms": round(self.max * 1000, 2), "gc_gen2": self.gen2}
        self.max = 0.0
        return out


class GilProbe:
    """A thread that sleeps 5 ms at a time and notes how late it wakes: how long the GIL (or the OS)
    held Python up, per second. About 0.1 % of a core."""

    PERIOD = 0.005

    def __init__(self):
        self.max = 0.0
        self._stop = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name="gil-probe", daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(1)

    def _run(self):
        while not self._stop.is_set():
            t = time.perf_counter()
            time.sleep(self.PERIOD)
            self.max = max(self.max, time.perf_counter() - t - self.PERIOD)

    def second(self) -> dict:
        out = {"gil_probe_max_ms": round(self.max * 1000, 2)}
        self.max = 0.0
        return out
