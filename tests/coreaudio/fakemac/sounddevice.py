"""A stand-in for sounddevice and CoreAudio, so tests/coreaudio runs where there is no Mac.

`run_all.py --fake` puts this folder first on PYTHONPATH and points SYNCAUDIO_FAKE_MAC at a folder,
so every process it starts - recorder, engine_play, syncaudio.py itself - imports this instead of the
real sounddevice. Its devices are fake BlackHoles: libsacore's test IO thread (sa_fake_loop: real
time, hard deadlines) on a tape each process maps from that folder, so what one process plays another
records, sample-aligned, like BlackHole's loopback. It sizes buffers the way PortAudio v19.7.0 does
(so 1.1.6's settings get their tiny buffer here too), and stands in for sacore.mac: CoreAudio's
facts, its listeners, and the default output and sample rates, kept in that folder so every process
sees a change.
"""
import ctypes
import json
import mmap
import os
import types

import numpy as np

import sacore

__version__ = "0.5.6"
_libname = None             # no PortAudio: sacore.mac below answers stream_device itself
RATE = 48000
TAPE = 1 << 20              # frames of loopback per device (22 s)
RANGE = (15, 4096)          # the IO buffer sizes a device allows (kAudioDevicePropertyBufferFrameSizeRange)
STATE = os.environ.get("SYNCAUDIO_FAKE_MAC") or "/tmp/syncaudio-fake-mac"
# name, CoreAudio id, safety offset and stream latency (frames): BlackHole, and the CI's two twins
DEVICES = [("BlackHole 2ch", 81, 0, 0)]
if os.environ.get("SYNCAUDIO_FAKE_TWINS") == "1":
    DEVICES += [("SyncTestBT 2ch", 82, 3840, 3840), ("SyncTestSpk 2ch", 83, 0, 0)]
_buffers: dict = {}         # (device, "output" | "input") -> frames of the last stream opened here
_tapes: dict = {}


class _FFI:
    @staticmethod
    def cast(_type, value):
        return value


_ffi = _FFI()


def _initialize():
    pass


def _terminate():
    pass


def get_portaudio_version():
    return 1246976, "PortAudio V19.7.0-devel (tests/coreaudio fake)"


def _state(name: str, default: dict) -> dict:
    try:
        with open(os.path.join(STATE, name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _save(name: str, value: dict):
    os.makedirs(STATE, exist_ok=True)
    tmp = os.path.join(STATE, f".{name}.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f)
    os.replace(tmp, os.path.join(STATE, name))


def _info(i: int) -> dict:
    name, _id, safety, strlat = DEVICES[i]
    low, high = (safety + strlat + 64) / RATE, (safety + strlat + 512) / RATE
    return {"name": name, "index": i, "hostapi": 0, "max_input_channels": 2, "max_output_channels": 2,
            "default_low_input_latency": low, "default_low_output_latency": low,
            "default_high_input_latency": high, "default_high_output_latency": high, "default_samplerate": RATE}


def _index(device) -> int:
    if isinstance(device, int):
        if 0 <= device < len(DEVICES):
            return device
    else:
        for match in (lambda n: n == device, lambda n: device.lower() in n.lower()):
            hits = [i for i, d in enumerate(DEVICES) if match(d[0])]
            if hits:
                return hits[0]
    raise ValueError(f"No device {device!r}")


def query_devices(device=None, kind=None):
    if device is None:
        return [_info(i) for i in range(len(DEVICES))]
    return _info(_index(device))


class _Default:
    @property
    def device(self):
        name = _state("default_output", {"name": DEVICES[0][0]})["name"]
        return [0, next((i for i, d in enumerate(DEVICES) if d[0] == name), 0)]


default = _Default()


def _tape(i: int) -> int:
    """The address of device i's loopback tape, shared with every other process."""
    if i not in _tapes:
        os.makedirs(STATE, exist_ok=True)
        size = TAPE * 2 * 4
        fd = os.open(os.path.join(STATE, f"tape-{DEVICES[i][1]}"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if os.fstat(fd).st_size < size:
                os.ftruncate(fd, size)
            mm = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        _tapes[i] = (mm, ctypes.addressof(ctypes.c_char.from_buffer(mm)))     # mapped for good
    return _tapes[i][1]


class _Stream:
    """A stream on a fake BlackHole, with PortAudio's buffer size: what was asked for, or with no
    blocksize max(latency x rate - the device's fixed latency, 1) clipped to the device's range."""

    def __init__(self, kind, device, channels, blocksize, latency, callback, userdata):
        if channels != 2:
            raise ValueError("the fake BlackHoles are stereo")
        self.kind, self.device, self.channels = kind, _index(device), channels
        _name, self.id, self.safety, strlat = DEVICES[self.device]
        fixed = self.safety + strlat
        self.blocksize = blocksize or 0
        self.frames = blocksize or min(RANGE[1], max(RANGE[0], int(latency * RATE) - fixed, 1))
        self.latency = (self.frames + fixed) / RATE
        self.callback, self.userdata = callback, userdata
        self.fake = self.stats = None
        self.active = False
        _buffers[(self.device, kind)] = self.frames

    def start(self):
        self.fake = sacore.lib.sa_fake_loop(0 if self.kind == "output" else 1, self.callback, self.userdata,
                                            self.frames, self.channels, RATE, self.safety, _tape(self.device),
                                            TAPE, 1)
        if not self.fake:
            raise OSError("sa_fake_loop failed")
        self.active = True

    def abort(self, ignore_errors=True):
        if self.fake:
            stats = (ctypes.c_uint64 * 6)()
            t0, maxcb = ctypes.c_double(), ctypes.c_double()
            sacore.lib.sa_fake_stop(self.fake, stats, ctypes.byref(t0), ctypes.byref(maxcb))
            sacore.lib.sa_fake_free(self.fake)
            self.stats = dict(zip(("cycles", "late", "skipped", "forced", "late_wake", "late_cb"), stats))
            self.fake = None
        self.active = False

    stop = close = abort


def _StreamBase(kind, samplerate=None, blocksize=None, device=None, channels=None, dtype=None, latency=None,
                callback=None, userdata=None, wrap_callback=None, **_):
    return _Stream(kind, device, channels, blocksize, latency or 0.0, callback, userdata)


class _Flags(int):
    output_underflow = property(lambda self: bool(self & 4))
    input_overflow = property(lambda self: bool(self & 2))
    _flags = property(int)


def OutputStream(samplerate=None, device=None, channels=None, dtype=None, latency=None, callback=None,
                 blocksize=0, **_):
    """A Python callback, entered from the IO thread through a ctypes trampoline that takes the GIL
    first, as cffi's does for the real sounddevice."""
    def tramp(_i, o, frames, ti, flags, _user):
        out = np.ctypeslib.as_array((ctypes.c_float * (frames * channels)).from_address(o))
        t = ti.contents
        callback(out.reshape(frames, channels), frames,
                 types.SimpleNamespace(currentTime=t.currentTime, outputBufferDacTime=t.outputBufferDacTime,
                                       inputBufferAdcTime=t.inputBufferAdcTime), _Flags(flags))
        return 0
    c = sacore.CALLBACK(tramp)
    st = _Stream("output", device, channels, blocksize, latency or 0.0, ctypes.cast(c, ctypes.c_void_p).value, None)
    st.keep = c
    return st


# ------------------------------------------------------------------ CoreAudio's side (sacore.mac)
class _Counter:
    def __init__(self, read):
        self._read = read

    @property
    def value(self) -> int:
        return self._read()

    def close(self):
        pass


class FakeMac:
    """sacore.mac for the fake devices. The default output and each device's nominal rate live in
    the shared folder (a change bumps a count, which is what the listeners report)."""

    SEL = {k: int.from_bytes(v.encode(), "big") for k, v in (
        ("overload", "over"), ("latency", "ltnc"), ("nominal_rate", "nsrt"), ("alive", "livn"),
        ("default_output", "dOut"), ("global", "glob"), ("output", "outp"), ("wildcard", "****"),
        ("bluetooth", "blue"), ("bluetooth_le", "blea"), ("virtual", "virt"))}
    system = 1

    def _rate(self, i):
        return _state(f"rate-{DEVICES[i][1]}", {"rate": float(RATE), "changes": 0})

    def device_info_id(self, id, output=True):
        i = next((i for i, d in enumerate(DEVICES) if d[1] == id), None)
        if i is None:
            return None
        name, _id, safety, strlat = DEVICES[i]
        rate = self._rate(i)["rate"]
        return {"id": id, "transport": "virt", "alive": 1, "out_channels": 2, "in_channels": 2, "latency": 0,
                "safety": safety, "stream_latency": strlat,
                "buffer_frames": _buffers.get((i, "output" if output else "input"), 512),
                "buffer_min": RANGE[0], "buffer_max": RANGE[1], "nominal_rate": rate, "actual_rate": rate,
                "uid": name.replace(" ", "") + "_UID", "name": name}

    def device_info(self, name, output=True):
        try:
            return self.device_info_id(DEVICES[_index(name)][1], output)
        except ValueError:
            return None

    def stream_device(self, sd, stream, output=True):
        return DEVICES[stream.device][1]

    def listen(self, id, sel, scope):
        if id == self.system and sel == self.SEL["default_output"]:
            return _Counter(lambda: _state("default_output", {"changes": 0})["changes"])
        i = next((i for i, d in enumerate(DEVICES) if d[1] == id), None)
        if i is not None and sel == self.SEL["nominal_rate"]:
            return _Counter(lambda: self._rate(i)["changes"])
        return _Counter(lambda: 0)

    def default_output(self):
        return DEVICES[default.device[1]][1]

    def set_default_output(self, id):
        info = self.device_info_id(id)
        if not info:
            return False
        _save("default_output", {"name": info["name"],
                                 "changes": _state("default_output", {"changes": 0})["changes"] + 1})
        return True

    def set_nominal_rate(self, id, rate):
        i = next((i for i, d in enumerate(DEVICES) if d[1] == id), None)
        if i is None:
            return False
        _save(f"rate-{id}", {"rate": float(rate), "changes": self._rate(i)["changes"] + 1})
        return True

    def qos_interactive(self):
        return True


sacore.mac = FakeMac()
