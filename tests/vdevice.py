"""A virtual audio device for the engine tests: PortAudio's and the HAL's part, in virtual time.

It calls libsacore's real C callbacks (through ctypes, the way PortAudio calls the function pointer)
with made-up DAC / ADC times, so a minute of audio takes about a second and anything macOS does can be
made to happen on cue: wobbling or missing timestamps, a device clock that runs fast, IO cycles the HAL
skips, losses smaller than a buffer, latency steps. Each simulated Mac has its own clock (`local`); the
event loop runs on true time, which is also the host's clock.

What an output device plays goes onto a tape by TRUE ear time (slot k reaches the ear at
ear0 + k / true rate), and judge() checks it at the speaker: every frame's content against the content
due at that moment. The test signal (signal()) is a 997 Hz sine on the left, for continuity, and on the
right a ramp that says which sample it is, ((n mod 65536) / 32768 - 1) * 0.9, so timing reads to a
small fraction of a sample. The real-thread tests (sacore's fake IO thread) record the same kind of
tape and use the same judge.
"""
import ctypes
import heapq
import itertools
import math
import types

import numpy as np

import sacore

RATE = 48000
F_SINE = 997.0
BLOCK = 480                 # the host's capture block (10 ms)


class TimeInfo(ctypes.Structure):           # PaStreamCallbackTimeInfo
    _fields_ = [("inputBufferAdcTime", ctypes.c_double), ("currentTime", ctypes.c_double),
                ("outputBufferDacTime", ctypes.c_double)]


CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
                            ctypes.POINTER(TimeInfo), ctypes.c_ulong, ctypes.c_void_p)


def signal(n0: int, k: int) -> np.ndarray:
    n = np.arange(n0, n0 + k)
    pcm = np.empty((k, 2), np.float32)
    pcm[:, 0] = 0.5 * np.sin(2 * np.pi * F_SINE * n / RATE)
    pcm[:, 1] = ((n % 65536) / 32768 - 1) * 0.9
    return pcm


CODE = 32768


def code(k0: int, k: int, channels: int = 2) -> np.ndarray:
    """What an input device captures at device frames k0...: its own index, ((k mod 32768) - 16384) /
    32768, the same code as sacore's fake input thread. Exact in float and in 16-bit PCM (the wire)."""
    v = ((np.arange(k0, k0 + k) % CODE - CODE // 2) / CODE).astype(np.float32)
    return np.repeat(v[:, None], channels, axis=1)


def decode(v) -> np.ndarray:
    """The frame index mod 32768 a captured value says (a zeroed frame reads as 16384)."""
    return np.rint(np.asarray(v, np.float64) * CODE).astype(np.int64) + CODE // 2


class Sim:
    """Events in true time. `t` is now."""

    def __init__(self, t: float = 1000.0):
        self.t = self.start = t
        self.q: list = []
        self.seq = itertools.count()

    def at(self, t: float, fn):
        heapq.heappush(self.q, (max(t, self.t), next(self.seq), fn))

    def every(self, period: float, fn, first: float | None = None, skip=lambda t: False):
        """fn() every `period` s (not while skip(t) - a stalled thread)."""
        def go():
            if not skip(self.t):
                fn()
            self.at(self.t + period, go)
        self.at(self.t if first is None else first, go)

    def run(self, seconds: float):
        """Until `seconds` after the start."""
        self.advance(self.start + seconds - self.t)

    def advance(self, seconds: float):
        end = self.t + seconds
        while self.q and self.q[0][0] <= end:
            self.t, _, fn = heapq.heappop(self.q)
            fn()
        self.t = end


class Device:
    """One output or input device on a ring, called back once per IO cycle in virtual time.

    IO cycle at device frame f happens at io0 + f / true_rate (true time). Output: its frames reach the
    ear `lead` later; input: they were sampled `lead` earlier. Its timestamps are in the Mac's own
    clock, `local(true time)`, plus report(k) (wobble), or 0 when invalid(k)."""

    def __init__(self, sim: Sim, ring, frames=1024, lead=0.2, ppm=0.0, seconds=30.0, local=lambda t: t,
                 report=lambda k: 0.0, invalid=lambda k: False):
        self.sim, self.ring, self.local, self.report, self.invalid = sim, ring, local, report, invalid
        self.out = ring.kind == "output"
        self.frames = itertools.cycle(frames) if isinstance(frames, (list, tuple)) else itertools.repeat(frames)
        self.true_rate = RATE * (1 + ppm * 1e-6)
        self.io0 = sim.t
        self.lead0 = self.lead = lead
        self.f = 0                      # device frames elapsed, skipped ones included
        self.k = 0                      # callbacks made
        self.cb = CALLBACK(ring.callback)
        n = int((seconds + 1) * self.true_rate) + 8192
        self.tape = np.empty((n, ring.channels), np.float32)
        self.tape.fill(0.0)             # touched now: a page fault inside the callback would count as its cost
        self._spare = np.zeros((8192, ring.channels), np.float32)
        self.dropped = []               # (true time, frames) the HAL skipped
        self._skip = 0
        self.stopped = False
        sim.at(self.io0, self._cycle)

    @property
    def ear0(self) -> float:
        return self.io0 + self.lead0

    def tau(self) -> np.ndarray:
        """True ear time of every tape slot."""
        return self.ear0 + np.arange(len(self.tape)) / self.true_rate

    def _cycle(self):
        if self.stopped:
            return
        if self._skip:                  # the HAL skips these frames: time passes, nobody is called
            self.dropped.append((self.sim.t, self._skip))
            self.f, self._skip = self.f + self._skip, 0
            self.sim.at(self.io0 + self.f / self.true_rate, self._cycle)
            return
        n = next(self.frames)
        t_io = self.io0 + self.f / self.true_rate
        stamp = t_io + self.lead if self.out else t_io - self.lead
        stamp = 0.0 if self.invalid(self.k) else self.local(stamp) + self.report(self.k)
        ti = TimeInfo(0.0 if self.out else stamp, self.local(t_io), stamp if self.out else 0.0)
        if self.out:
            slot = self.f + round((self.lead - self.lead0) * self.true_rate)
            if 0 <= slot and slot + n <= len(self.tape):
                buf = self.tape[slot:slot + n]
            else:
                buf = self._spare[:n]
            self.cb(None, buf.ctypes.data, n, ctypes.byref(ti), 0, self.ring.h)
        else:
            pcm = code(self.f, n, self.ring.channels)
            self.cb(pcm.ctypes.data, None, n, ctypes.byref(ti), 0, self.ring.h)
        self.f += n
        self.k += 1
        self.sim.at(self.io0 + self.f / self.true_rate, self._cycle)

    def drop(self, frames: int):
        """The HAL skips the next `frames` of IO (whole cycles: an overload; less: a loss under the C
        side's threshold)."""
        self._skip += frames

    def stop(self):
        self.stopped = True

    def latency_step(self, seconds: float):
        """The device's latency changes (a Bluetooth renegotiation): its frames reach the ear this much
        later from now on, and PortAudio's DAC times say so."""
        self.lead += seconds


class Feeder:
    """The host: blocks of BLOCK samples captured from T0 on (host clock = true time), each stamped
    play_at = T0 + n / RATE + delay(n) and handed to sb.feed once it arrives: net after its last
    sample was captured, or when a network stall ends (TCP keeps the order). `local` is the receiving
    Mac's clock (the stamps stay on the host's)."""

    def __init__(self, sim: Sim, sb, T0: float, delay=lambda n: 0.5, net=0.005, stalls=(), content=signal):
        self.sim, self.sb, self.T0, self.delay, self.net = sim, sb, T0, delay, net
        self.stalls, self.content = stalls, content
        self.k, self.last = 0, 0.0
        self.stamps = []
        sim.at(self._arrival(0), self._deliver)

    def stamp(self, n: int) -> float:
        return self.T0 + n / RATE + self.delay(n)

    def _arrival(self, k: int) -> float:
        t = self.T0 + (k + 1) * BLOCK / RATE + self.net
        for t0, dur in self.stalls:
            if t0 <= t < t0 + dur:
                t = t0 + dur
        self.last = max(self.last, t)
        return self.last

    def _deliver(self):
        n = self.k * BLOCK
        self.stamps.append((n, self.stamp(n)))
        self.sb.feed(n, self.stamp(n), self.content(n, BLOCK))
        self.k += 1
        self.sim.at(self._arrival(self.k), self._deliver)


def steady(T0: float, D: float):
    """due() for a constant delay D: the content index due at host time s."""
    return lambda s: ((s - T0 - D) * RATE, np.zeros(len(s), bool))


def judge(tape, tau, due, t_from: float, t_to: float = math.inf, allow=(), merge=480) -> dict:
    """What reached the ear, frame by frame, from t_from to t_to (true time).

    due(s) -> (content index due at host times s, in_gap): in_gap marks silence a delay increase put
    there. A frame is bad if it is silent where music is due (a hole), not silent in a gap, decodes to
    a jump in content (|dn - dn_due| > 0.05 sample:
    float32 resampling reads the ramp to ~0.01), or isn't the clean sine for the content it decodes
    to (a fade, a click, a resampling error). Bad frames closer than `merge` make one glitch. `allow`:
    (t0, t1) stretches where a deliberate change may be heard. err: decoded minus due content, in
    seconds, over the good frames (+ = late)."""
    sel = np.nonzero((tau >= t_from) & (tau < t_to))[0]
    s = tau[sel]
    y = tape[sel].astype(np.float64)
    n_due, gap = due(s)
    silent = (np.abs(y[:, 0]) < 1e-6) & (np.abs(y[:, 1]) < 1e-6)
    heard = (y[:, 1] / 0.9 + 1) * 32768                     # n mod 65536 as heard
    n_dec = n_due + (heard - n_due + 32768) % 65536 - 32768
    # The ramp's own jump, which the resampler smears over +-16 samples into values that look like any
    # other position: find it from where the content is (due + the typical error in each 1024 frames)
    off = np.zeros(len(sel))
    for i in range(0, len(sel), 1024):
        off[i:i + 1024] = np.median((n_dec - n_due)[i:i + 1024])
    at = (n_due + off) % 65536
    wrap = (at < 48) | (at > 65536 - 48)
    wave = np.abs(y[:, 0] - 0.5 * np.sin(2 * np.pi * F_SINE * n_dec / RATE)) > 1e-3
    jump = np.zeros(len(sel), bool)
    jump[1:] = (np.abs(np.diff(n_dec) - np.diff(n_due)) > 0.05) & ~wrap[1:] & ~wrap[:-1] & ~gap[1:] & ~gap[:-1]
    bad = np.where(gap, ~silent, silent | (~wrap & (wave | jump)))
    ok = np.ones(len(sel), bool)
    for t0, t1 in allow:
        ok &= ~((s >= t0) & (s < t1))
    bad &= ok
    good = ok & ~gap & ~silent & ~wrap & ~wave
    err = (n_dec[good] - n_due[good]) / RATE
    g = np.nonzero(good)[0]                 # content samples per device frame, over 10 ms of good frames
    span = np.nonzero(g[480:] - g[:-480] == 480)[0] if len(g) > 480 else np.array([], int)
    speed = (n_dec[g[span + 480]] - n_dec[g[span]]) / 480
    where = np.nonzero(bad)[0]
    runs = int(len(where) > 0) + int(np.count_nonzero(np.diff(where) > merge))
    starts = s[where[np.r_[True, np.diff(where) > merge]]] if len(where) else np.array([])
    a = np.abs(err) * 1000
    return {"glitches": runs, "glitch_at": [round(float(t), 3) for t in starts[:10]],
            "holes_ms": round(float(np.count_nonzero(silent & ~gap & ok)) / RATE * 1000, 2),
            "err_max_ms": round(float(a.max()), 4) if len(a) else None,
            "err_p99_ms": round(float(np.percentile(a, 99)), 4) if len(a) else None,
            "err_median_ms": round(float(np.median(err)) * 1000, 4) if len(a) else None,
            "judged_s": round(len(sel) / RATE, 1), "good_s": round(float(good.sum()) / RATE, 1),
            "speed": [round(float(speed.min()), 5), round(float(speed.max()), 5)] if len(speed) else None,
            "err": err, "err_t": s[good]}


# ------------------------------------------------------------------------- fake sounddevice modules
class _Devices:
    """query_devices() over a list of device dicts (name, max_output_channels, max_input_channels,
    default_samplerate, default_low/high_output/input_latency) plus each one's simulated behaviour."""

    __version__ = "0.5.6"

    def __init__(self, devices):
        self.devices = [{"max_output_channels": 0, "max_input_channels": 0, "default_samplerate": RATE,
                         "default_low_output_latency": 0.01, "default_high_output_latency": 0.1,
                         "default_low_input_latency": 0.01, "default_high_input_latency": 0.1, **d}
                        for d in devices]
        self.opened = []                # every _StreamBase call, as given (what PortAudio is handed)

    def _index(self, device):
        if isinstance(device, int):
            return device
        return next(i for i, d in enumerate(self.devices) if d["name"] == device)

    def query_devices(self, device=None, kind=None):
        if device is None:
            return [dict(d, index=i) for i, d in enumerate(self.devices)]
        i = self._index(device)
        return dict(self.devices[i], index=i)

    def get_portaudio_version(self):
        return 1246976, "PortAudio V19.7.0-devel (fake)"

    def _StreamBase(self, kind, samplerate, blocksize, device, channels, dtype, latency, callback,
                    userdata, wrap_callback):
        self.opened.append(dict(kind=kind, samplerate=samplerate, blocksize=blocksize, device=device,
                                channels=channels, dtype=dtype, latency=latency, callback=callback,
                                userdata=userdata, wrap_callback=wrap_callback))
        return self._stream(kind, self._index(device), channels, blocksize, callback, userdata)


class VirtualSD(_Devices):
    """sounddevice for AudioOutput / AudioInput in virtual time: a started stream runs a Device on
    the C callback it was given, with the buffer size asked for. Device keys: lead, ppm, frames (the
    size when none is asked for), force_frames (a device that ignores the request), seconds, and
    local / report (the Mac's clock and the timestamps' wobble, as Device takes them)."""

    def __init__(self, sim: Sim, devices):
        super().__init__(devices)
        self.sim = sim
        self.streams = []

    def _stream(self, kind, i, channels, blocksize, callback, userdata):
        st = VirtualStream(self, kind, i, channels, blocksize, callback, userdata)
        self.streams.append(st)
        return st


class PortAudioError(Exception):
    pass


class VirtualStream:
    def __init__(self, sd, kind, i, channels, blocksize, callback, userdata):
        self.sd, self.kind, self.profile = sd, kind, sd.devices[i]
        self.ring = types.SimpleNamespace(kind=kind, channels=channels, callback=callback, h=userdata)
        self.blocksize, self.device = blocksize, None
        self.latency = self.profile.get("lead", 0.2)
        self.active = self.closed = False
        self.stuck = False              # PortAudio can't stop it: it keeps calling back

    def start(self):
        p = self.profile
        self.device = Device(self.sd.sim, self.ring, frames(p, self.blocksize), p.get("lead", 0.2),
                             p.get("ppm", 0.0), p.get("seconds", 30.0), local=p.get("local", lambda t: t),
                             report=p.get("report", lambda k: 0.0))
        self.active = True

    def _refused(self, what, ignore_errors):
        """A stuck stream refuses to stop; like sounddevice, PortAudio's error is raised only if asked."""
        if self.stuck and not ignore_errors:
            raise PortAudioError(f"Error {what} stream: Bad stream pointer [PaErrorCode -9988]")
        return self.stuck

    def abort(self, ignore_errors=True):
        if self._refused("aborting", ignore_errors):
            return
        if self.device:
            self.device.stop()
        self.active = False

    def close(self, ignore_errors=True):
        if not self._refused("closing", ignore_errors):     # Pa_CloseStream closes only what it could stop
            self.closed = True


def frames(profile, blocksize):
    """The buffer size a stream gets: a device may ignore what was asked for."""
    return profile.get("force_frames") or blocksize or profile.get("frames", 512)


class _Flags(int):
    """sounddevice.CallbackFlags, as much as 1.1.6 uses."""
    output_underflow = property(lambda self: bool(self & 4))
    input_overflow = property(lambda self: bool(self & 2))
    _flags = property(int)


class FakeSD(_Devices):
    """sounddevice on sacore's fake IO thread: real time, hard deadlines (a late callback loses its
    buffer and the cycles it overran are skipped, as on macOS). _StreamBase hands it the C callback.
    OutputStream (1.1.6's Python callback) goes through a ctypes trampoline that has to take the GIL
    first, as cffi's does for sounddevice. Device keys: lead, frames / force_frames (as VirtualSD), ppm,
    wobble, skip_every (HAL-side skips every so many s), record (seconds of output kept)."""

    def _stream(self, kind, i, channels, blocksize, callback, userdata):
        return FakeStream(self.devices[i], kind, channels, blocksize, callback, userdata)

    def OutputStream(self, samplerate, device, channels, dtype, latency, callback, blocksize=0, **_):
        self.opened.append(dict(kind="output", samplerate=samplerate, blocksize=blocksize, device=device,
                                channels=channels, latency=latency, callback=callback, wrap_callback="array"))

        def tramp(i, o, frames, ti, flags, user):
            out = np.ctypeslib.as_array((ctypes.c_float * (frames * channels)).from_address(o))
            t = ti.contents
            callback(out.reshape(frames, channels), frames,
                     types.SimpleNamespace(currentTime=t.currentTime, outputBufferDacTime=t.outputBufferDacTime,
                                           inputBufferAdcTime=t.inputBufferAdcTime), _Flags(flags))
            return 0
        c = CALLBACK(tramp)
        st = FakeStream(self.devices[self._index(device)], "output", channels, blocksize,
                        ctypes.cast(c, ctypes.c_void_p).value, None)
        st.keep = c
        return st


class FakeStream:
    def __init__(self, profile, kind, channels, blocksize, callback, userdata):
        self.profile, self.kind, self.channels = profile, kind, channels
        self.blocksize, self.callback, self.userdata = blocksize, callback, userdata
        self.latency = profile.get("lead", 0.2)
        self.fake = self.stats = self.rec = None
        self.active = False

    def start(self):
        p = self.profile
        self.frames = frames(p, self.blocksize)
        self.rec_frames = int(p.get("record", 0) * RATE) + 8192 if p.get("record") else 0
        self.started = sacore.now()
        self.fake = sacore.lib.sa_fake_start(0 if self.kind == "output" else 1, self.callback, self.userdata,
                                             self.frames, self.channels, RATE, p.get("lead", 0.2), p.get("wobble", 0.0),
                                             p.get("ppm", 0.0), p.get("skip_every", 0.0), self.rec_frames, 1)
        if not self.fake:
            raise OSError("sa_fake_start failed")
        self.active = True

    def abort(self, ignore_errors=True):
        """Stops the fake IO thread; afterwards: stats, t0 (frame 0 reached the ear at t0 + lead), rec.
        Like sounddevice's, a stream that was stopped or closed is never `active` (a test may set it to
        play a device that stopped calling back while PortAudio still says it runs)."""
        if not self.fake:
            self.active = False
            return
        stats = (ctypes.c_uint64 * 6)()
        t0, maxcb = ctypes.c_double(), ctypes.c_double()
        sacore.lib.sa_fake_stop(self.fake, stats, ctypes.byref(t0), ctypes.byref(maxcb))
        self.stats = dict(zip(("cycles", "late", "skipped", "forced", "late_wake", "late_cb"), stats))
        self.stats["max_callback_ms"] = maxcb.value * 1000
        self.t0 = t0.value
        if self.rec_frames:
            self.rec = np.ctypeslib.as_array((ctypes.c_float * (self.rec_frames * self.channels)).from_address(
                sacore.lib.sa_fake_rec(self.fake))).reshape(self.rec_frames, self.channels).copy()
        sacore.lib.sa_fake_free(self.fake)
        self.fake = None
        self.active = False

    def stop(self, ignore_errors=True):
        self.abort()

    def close(self, ignore_errors=True):
        self.abort()

    def tau(self) -> np.ndarray:
        """True ear time of each recorded frame."""
        true = RATE * (1 + self.profile.get("ppm", 0.0) * 1e-6)
        return self.t0 + self.profile.get("lead", 0.2) + np.arange(len(self.rec)) / true

    def live(self, seconds: float) -> np.ndarray:
        """While it runs: the last `seconds` it played, roughly (the fake starts 50 ms after start())."""
        if not self.fake:
            return self.rec[:0] if self.rec is None else self.rec[-int(seconds * RATE):]
        rec = np.ctypeslib.as_array((ctypes.c_float * (self.rec_frames * self.channels)).from_address(
            sacore.lib.sa_fake_rec(self.fake))).reshape(self.rec_frames, self.channels)
        end = min(self.rec_frames, max(0, int((sacore.now() - self.started - 0.05) * RATE) - self.frames))
        return rec[max(0, end - int(seconds * RATE)):end].copy()
