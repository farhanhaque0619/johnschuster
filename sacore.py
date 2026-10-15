"""libsacore (native/sacore.c) from Python.

Python never runs on CoreAudio's IO threads: the C callbacks there only copy between the device
and the rings below. Python talks to the rings from ordinary threads - the output engine writes
audio 150 ms to 1 s ahead of the DAC, the capture engine drains BlackHole.

The ring, content and resampler calls take microseconds and never wait, so they keep the GIL
(ctypes.PyDLL): letting go of it and getting it back costs a wait for every other busy Python thread,
and on macOS those waits add up until the output engine can't keep up (CI saw it starve for seconds
under load). Everything that may block - CoreAudio property calls, the tests' fake IO thread - goes
through ctypes.CDLL, which lets go of the GIL while the C code runs.

Also here: the ring of received audio and its resampler (Content), and on macOS the optional
helpers in sacore_mac.c (`mac`, always called through mac_call so a failure there can't stop audio).
"""
from __future__ import annotations

import ctypes
import math
import os
import sys

import numpy as np

LOG_DTYPE = np.dtype([("seq", "<u8"), ("pos", "<u8"), ("dev_time", "<f8"), ("current_time", "<f8"),
                      ("entry_time", "<f8"), ("exit_time", "<f8"), ("frames", "<u4"), ("valid", "<u4"),
                      ("skipped", "<u4"), ("flags", "<u4")])      # sa_log in sacore.c
assert LOG_DTYPE.itemsize == 64
LOG_N = 4096
STATE = ("r", "w", "s", "callbacks", "underrun_frames", "underrun_events", "skips", "skipped_frames",
         "overflows", "min_lead_frames", "log_seq", "flag_or", "max_cb_ns")
NO_LEAD = 2 ** 64 - 1           # min_lead_frames when no callback ran since the last state()


def _candidates():
    if os.environ.get("SACORE_LIB"):
        return [os.environ["SACORE_LIB"]]       # exactly this one (tests, CI, the app build)
    base = getattr(sys, "_MEIPASS", None)       # inside the packaged app
    here = os.path.dirname(os.path.abspath(__file__))
    return ([os.path.join(base, "libsacore.dylib")] if base else []) + \
        [os.path.join(here, name) for name in ("libsacore.dylib", "libsacore.so")]


def _bind(lib):
    u32, u64, i64, dbl, ptr = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int64, ctypes.c_double, ctypes.c_void_p

    def fn(name, restype, *argtypes):
        f = getattr(lib, name)
        f.restype, f.argtypes = restype, list(argtypes)

    fn("sa_version", u32)
    fn("sa_now", dbl)
    fn("sa_ring_new", ptr, u32, u32, dbl, u32)
    fn("sa_ring_free", None, ptr)
    fn("sa_ring_state", None, ptr, ptr)
    fn("sa_ring_log", u32, ptr, u64, ptr, u32)
    fn("sa_out_write", i64, ptr, u64, ptr, u32, u32)
    fn("sa_out_cut", ctypes.c_int, ptr, u64, u32)
    fn("sa_in_read", i64, ptr, u64, ptr, u32)
    fn("sa_ring_debug_stall", None, ptr, u32)
    fn("sa_content_new", ptr, u32, u32)
    fn("sa_content_free", None, ptr)
    fn("sa_content_put", i64, ptr, i64, ptr, u32)
    fn("sa_content_range", None, ptr, ptr)
    fn("sa_set_filter", ctypes.c_int, ptr, u32, u32)
    fn("sa_resample", u32, ptr, dbl, dbl, u32, ptr, ctypes.c_float, ctypes.c_float, ctypes.c_int)
    if hasattr(lib, "sa_fake_start"):           # the tests' fake IO thread (not in the shipped build)
        fn("sa_fake_start", ptr, ctypes.c_int, ptr, ptr, u32, u32, dbl, dbl, dbl, dbl, dbl, u64, ctypes.c_int)
        fn("sa_fake_stop", None, ptr, ptr, ptr, ptr)
        fn("sa_fake_policy", ctypes.c_int, ptr)
        fn("sa_fake_rec", ptr, ptr)
        fn("sa_fake_free", None, ptr)
        if hasattr(lib, "sa_fake_loop"):        # a fake BlackHole shared by processes (tests/coreaudio)
            fn("sa_fake_loop", ptr, ctypes.c_int, ptr, ptr, u32, u32, dbl, u32, ptr, u64, ctypes.c_int)
    if hasattr(lib, "sa_device_info"):          # sacore_mac.c
        fn("sa_device_info", ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ptr)
        fn("sa_device_info_id", ctypes.c_int, u32, ctypes.c_int, ptr)
        fn("sa_counter_new", ptr)
        fn("sa_counter_get", u64, ptr)
        fn("sa_counter_free", None, ptr)
        fn("sa_listen", ctypes.c_int, u32, u32, u32, ptr)
        fn("sa_unlisten", ctypes.c_int, u32, u32, u32, ptr)
        fn("sa_default_output", u32)
        fn("sa_set_default_output", ctypes.c_int, u32)
        fn("sa_set_nominal_rate", ctypes.c_int, u32, dbl)
        fn("sa_set_thread_qos_interactive", ctypes.c_int)
        fn("sa_selectors", None, ptr)


FAST = ("sa_now", "sa_version", "sa_ring_state", "sa_ring_log", "sa_out_write", "sa_out_cut", "sa_in_read",
        "sa_ring_debug_stall", "sa_content_put", "sa_content_range", "sa_resample")   # quick, never wait


def _load():
    for path in _candidates():
        if not os.path.exists(path):
            continue
        try:
            lib, fast = ctypes.CDLL(path), ctypes.PyDLL(path)
            _bind(lib)
            _bind(fast)
        except (OSError, AttributeError):
            continue
        for name in FAST:               # the same symbols, called without letting go of the GIL
            setattr(lib, name, getattr(fast, name))
        return lib, path
    return None, None


lib, LIBRARY = _load()
VERSION = lib.sa_version() if lib else None


def available() -> bool:
    return lib is not None


def now() -> float:
    """sacore's clock: the one PortAudio's timestamps use, == time.monotonic()."""
    return lib.sa_now()


def userdata(sd, h):
    """The ring handle as sounddevice wants a stream's userdata (a cffi pointer)."""
    return sd._ffi.cast("void *", h) if hasattr(sd, "_ffi") else h


class Lapped(Exception):
    """Capture fell a whole ring behind: those frames were overwritten before they were read."""


def _pcm(pcm, channels):
    pcm = np.ascontiguousarray(pcm, np.float32)
    if pcm.ndim != 2 or pcm.shape[1] != channels:
        raise ValueError(f"audio must be (frames, {channels}), not {pcm.shape}")
    return pcm


class Closed(ValueError):
    """A ring or content used after close(): its C memory is gone (C would crash)."""


def _live(h):
    if not h:
        raise Closed("sacore: used after close()")
    return h


class Ring:
    """One output or input ring (sacore.c). `callback` is the address of its C callback, to give
    PortAudio with `userdata(sd, ring.h)`. Close it only after the stream is closed."""

    def __init__(self, kind: str, channels: int, rate: float, seconds: float = 4.0):
        self.kind, self.channels, self.rate = kind, channels, rate
        self.cap = 1 << max(12, math.ceil(math.log2(rate * seconds)))
        self.h = lib.sa_ring_new(0 if kind == "output" else 1, channels, rate, self.cap)
        if not self.h:
            raise ValueError(f"sacore can't make a {kind} ring of {channels} channels")
        fn = lib.sa_out_callback if kind == "output" else lib.sa_in_callback
        self.callback = ctypes.cast(fn, ctypes.c_void_p).value
        self._state = (ctypes.c_uint64 * len(STATE))()
        self._log = np.zeros(LOG_N, LOG_DTYPE)
        self.log_seq = 0

    def state(self) -> dict:
        lib.sa_ring_state(_live(self.h), self._state)
        return dict(zip(STATE, self._state))

    def new_records(self) -> np.ndarray:
        """Callback records since the last call, oldest first (the newest LOG_N at most)."""
        n = lib.sa_ring_log(_live(self.h), self.log_seq, self._log.ctypes.data, LOG_N)
        recs = self._log[:n].copy()
        if n:
            self.log_seq = int(recs["seq"][-1])
        return recs

    def write(self, pos: int, pcm: np.ndarray, guard: int) -> int:
        """Output: audio for device frames pos... (see sa_out_write). Returns len(pcm), or -1."""
        pcm = _pcm(pcm, self.channels)
        return lib.sa_out_write(_live(self.h), pos, pcm.ctypes.data, len(pcm), guard)

    def cut(self, pos: int, guard: int) -> bool:
        """Output: the ring's audio ends at pos from now on (see sa_out_cut)."""
        return lib.sa_out_cut(_live(self.h), pos, guard) == 0

    def read(self, pos: int, n: int):
        """Input: device frames [pos, pos + n), or None if not all captured yet. Raises Lapped."""
        out = np.empty((n, self.channels), np.float32)
        got = lib.sa_in_read(_live(self.h), pos, out.ctypes.data, n)
        if got == -2:
            raise Lapped(pos)
        return out if got >= 0 else None

    def debug_stall(self, us: int):
        """CI only: make the next callback `us` microseconds late, once."""
        lib.sa_ring_debug_stall(_live(self.h), us)

    def close(self):
        if self.h:
            lib.sa_ring_free(self.h)
            self.h = None


class TimeInfo(ctypes.Structure):           # PortAudio's PaStreamCallbackTimeInfo
    _fields_ = [("inputBufferAdcTime", ctypes.c_double), ("currentTime", ctypes.c_double),
                ("outputBufferDacTime", ctypes.c_double)]


CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,      # PaStreamCallback
                            ctypes.POINTER(TimeInfo), ctypes.c_ulong, ctypes.c_void_p)


def smoke_test() -> bool:
    """Write a ramp into an output ring and call its C callback exactly as PortAudio does: the ramp
    must come out unchanged. The app's self-test runs this, so a libsacore whose arguments don't
    line up with this file's (or with PortAudio's) never ships."""
    ring = Ring("output", 2, 48000)
    try:
        ramp = (np.arange(2048, dtype=np.float32).reshape(1024, 2) - 1024) / 1024
        out = np.full((512, 2), np.nan, np.float32)
        wrote = ring.write(0, ramp, 0)
        CALLBACK(ring.callback)(None, out.ctypes.data, 512, ctypes.byref(TimeInfo(0.0, 99.99, 100.0)), 0, ring.h)
        st = ring.state()
        return wrote == 1024 and bool(np.array_equal(out, ramp[:512])) and (st["r"], st["callbacks"]) == (512, 1)
    finally:
        ring.close()


def make_filter(taps: int = 32, phases: int = 256, beta: float = 9.0) -> np.ndarray:
    """Kaiser-windowed sinc, one row per fractional phase plus the next whole sample; row 0 is an
    exact unit impulse, so playing at speed 1.0 on a whole sample changes nothing at all."""
    x = np.arange(taps)[None, :] - (taps // 2 - 1) - (np.arange(phases + 1)[:, None] / phases)
    win = np.i0(beta * np.sqrt(np.clip(1 - (x / (taps / 2)) ** 2, 0, 1))) / np.i0(beta)
    h = np.sinc(x) * win
    whole = x == np.rint(x)
    h[whole] = x[whole] == 0            # sinc is exactly 0 at whole samples, not numpy's 1e-17
    return np.ascontiguousarray(h / h.sum(axis=1, keepdims=True), np.float32)


FILTER = make_filter()
if lib is not None and lib.sa_set_filter(FILTER.ctypes.data, FILTER.shape[0] - 1, FILTER.shape[1]) != 0:
    raise RuntimeError("sacore refused the resampling filter")


class Content:
    """Received audio by stream sample index, in C memory (sa_content_put / sa_resample)."""

    def __init__(self, channels: int, rate: float, seconds: float = 8.0):
        self.channels, self.rate = channels, rate
        self.cap = 1 << max(13, math.ceil(math.log2(rate * seconds)))
        self.h = lib.sa_content_new(channels, self.cap)
        if not self.h:
            raise ValueError(f"sacore can't hold {channels} channels")
        self._range = (ctypes.c_int64 * 3)()

    def put(self, n: int, pcm: np.ndarray) -> int:
        """Store pcm at sample index n. Returns how many missing samples before it became silence."""
        pcm = _pcm(pcm, self.channels)
        filled = lib.sa_content_put(_live(self.h), n, pcm.ctypes.data, len(pcm))
        if filled < 0:
            raise ValueError(f"a block of {len(pcm)} samples doesn't fit")
        return filled

    def range(self):
        """(lo, end): the samples held, or (None, None) before the first put."""
        lib.sa_content_range(_live(self.h), self._range)
        lo, end, empty = self._range
        return (None, None) if empty else (lo, end)

    def resample(self, pos: float, ratio: float, n: int, dst: np.ndarray, g0=1.0, g1=1.0, add=False) -> int:
        """dst[:n] (+)= gain * content at pos, pos + ratio, ..., gain going g0 -> g1.
        Returns how many of the n frames had no content."""
        if (dst.dtype != np.float32 or not dst.flags.c_contiguous or dst.ndim != 2
                or dst.shape[1] != self.channels or len(dst) < n):
            raise ValueError(f"dst must be C-contiguous float32 (>= {n}, {self.channels})")
        return lib.sa_resample(_live(self.h), pos, ratio, n, dst.ctypes.data, g0, g1, int(add))

    def close(self):
        if self.h:
            lib.sa_content_free(self.h)
            self.h = None


# ------------------------------------------------------------------ macOS helpers (sacore_mac.c)
class DevInfo(ctypes.Structure):
    _fields_ = [(k, ctypes.c_uint32) for k in ("id", "transport", "alive", "out_channels", "in_channels",
                                                 "latency", "safety", "stream_latency", "buffer_frames",
                                                 "buffer_min", "buffer_max")] + \
               [("nominal_rate", ctypes.c_double), ("actual_rate", ctypes.c_double),
                ("uid", ctypes.c_char * 256), ("name", ctypes.c_char * 256)]


class _Mac:
    """CoreAudio's own facts and notifications. Use through mac_call(), never directly."""

    def __init__(self, lib):
        self.lib = lib
        sel = (ctypes.c_uint32 * 11)()
        lib.sa_selectors(sel)
        self.SEL = dict(zip(("overload", "latency", "nominal_rate", "alive", "default_output", "global",
                             "output", "wildcard", "bluetooth", "bluetooth_le", "virtual"), sel))
        self.system = 1                                 # kAudioObjectSystemObject

    def _info(self, d: DevInfo) -> dict:
        info = {k: getattr(d, k) for k, _ in DevInfo._fields_}
        info["uid"], info["name"] = d.uid.decode(errors="replace"), d.name.decode(errors="replace")
        info["transport"] = d.transport.to_bytes(4, "big").decode(errors="replace")    # 'blue', 'bltn', ...
        return info

    def device_info(self, name: str, output: bool = True):
        """Transport, latency / safety offset / stream latency in frames, the IO buffer in force and
        its range, rates, UID - for the device PortAudio calls `name`, in that direction."""
        d = DevInfo()
        return self._info(d) if self.lib.sa_device_info(name.encode(), int(output), ctypes.byref(d)) == 0 else None

    def device_info_id(self, id: int, output: bool = True):
        d = DevInfo()
        return self._info(d) if self.lib.sa_device_info_id(id, int(output), ctypes.byref(d)) == 0 else None

    def stream_device(self, sd, stream, output: bool = True):
        """The CoreAudio device id an open sounddevice stream really plays to (records from)."""
        try:
            pa = ctypes.CDLL(sd._libname)
            get = pa.PaMacCore_GetStreamOutputDevice if output else pa.PaMacCore_GetStreamInputDevice
        except (OSError, AttributeError, TypeError):     # not exported by this PortAudio: by name
            info = self.device_info(sd.query_devices(stream.device)["name"], output)
            return info["id"] if info else None
        get.restype, get.argtypes = ctypes.c_uint32, [ctypes.c_void_p]
        return get(int(sd._ffi.cast("uintptr_t", stream._ptr))) or None

    class Counter:
        """How many times macOS notified one property of one object since listen()."""

        def __init__(self, lib, id: int, sel: int, scope: int):
            self.lib, self.id, self.sel, self.scope = lib, id, sel, scope
            self.c = lib.sa_counter_new()
            if not self.c:
                raise OSError("no free sacore counter")
            err = lib.sa_listen(id, sel, scope, self.c)
            if err:
                lib.sa_counter_free(self.c)
                self.c = None
                raise OSError(f"AudioObjectAddPropertyListener: {err}")

        @property
        def value(self) -> int:
            return self.lib.sa_counter_get(self.c) if self.c else 0

        def close(self):
            if self.c:
                self.lib.sa_unlisten(self.id, self.sel, self.scope, self.c)
                self.lib.sa_counter_free(self.c)
                self.c = None

    def listen(self, id: int, sel: int, scope: int) -> "_Mac.Counter":
        return self.Counter(self.lib, id, sel, scope)

    def default_output(self):
        return self.lib.sa_default_output() or None

    def set_default_output(self, id: int) -> bool:
        return self.lib.sa_set_default_output(id) == 0

    def set_nominal_rate(self, id: int, rate: float) -> bool:
        return self.lib.sa_set_nominal_rate(id, rate) == 0

    def qos_interactive(self) -> bool:
        """Ask macOS to run the calling thread like the UI (performance cores, ahead of background work)."""
        return self.lib.sa_set_thread_qos_interactive() == 0


mac = _Mac(lib) if lib is not None and hasattr(lib, "sa_device_info") else None
_mac_failed: set = set()


def mac_call(name: str, *args, default=None):
    """mac.<name>(*args), or `default` if the helper is absent or fails (said once per helper).
    Nothing that plays audio depends on these: without them only facts and counts are lost."""
    fn = getattr(mac, name, None)
    if fn is None:
        return default
    try:
        return fn(*args)
    except Exception as e:      # noqa: BLE001 - any failure here must not reach the audio
        if name not in _mac_failed:
            _mac_failed.add(name)
            print(f"sacore.mac.{name} failed, carrying on without it: {e!r}", file=sys.stderr)
        return default
