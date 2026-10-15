#!/usr/bin/env python3
"""
syncaudio - play whatever one Mac is playing on several Macs, in sync.

The host Mac captures its system audio through BlackHole (a free virtual
audio device), stamps every block with the exact moment it should come out
of the speakers, and streams it over the local network. Every Mac - the host
included - holds each block until that moment, so all speakers play the same
sample at the same time.

    ./syncaudio host        # on the Mac that plays Spotify / YouTube / ...
    ./syncaudio join        # on your friend's Mac (finds the host by itself)
    ./syncaudio host --click    # metronome, to check the sync by ear
"""
from __future__ import annotations

import argparse
import errno
import json
import math
import os
import queue
import re
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np

import codec
import discovery

PORT = 47800                # TCP carries audio, UDP (same number) clock sync + discovery
MAGIC = "syncaudio/1"
BLOCK_MS = 10               # capture block size
OUTPUT_LATENCY = 0.1        # CoreAudio output buffer: roomy, so a busy moment in Python can't click;
                            # it's compensated, so it only adds to the delay
INPUT_LATENCY = 0.05        # same for capturing from BlackHole
APP_VERSION = "dev"         # the app sets its build version; exchanged so Macs can spot mismatches

# --------------------------------------------------------------------- wire
HELLO, HEADER, AUDIO, STATS, UPDATE, CONTROL = b"H", b"C", b"A", b"S", b"U", b"K"
OPUS = b"O"                         # like AUDIO, but an Opus packet instead of 16-bit samples
FRAME = struct.Struct("<cI")        # kind, payload length
AUDIO_HDR = struct.Struct("<qd")    # index of first sample, play_at (host clock)
PING = struct.Struct("<cId")        # b"P", seq, t1
PONG = struct.Struct("<cIddd")      # b"Q", seq, t1, t2 (host recv), t3 (host send)
DISCOVER = b"D" + MAGIC.encode()


def frame(kind: bytes, payload: bytes) -> bytes:
    return FRAME.pack(kind, len(payload)) + payload


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:])
        if not k:
            raise ConnectionError("connection closed")
        got += k
    return bytes(buf)


def recv_frame(sock: socket.socket):
    kind, n = FRAME.unpack(recv_exact(sock, FRAME.size))
    if n > 1 << 20:
        raise ConnectionError("garbled stream")
    return kind, recv_exact(sock, n)


# --------------------------------------------------------------- clock sync
class ClockSync:
    """Maps this Mac's clock onto the host's clock, NTP style.

    Pings carry (t1 local send, t2 host receive, t3 host send, t4 local
    receive). Wi-Fi delays are noisy and lopsided, so only the fastest
    round trips are trusted, and a line is fitted through them so the slow
    drift between the two clocks is followed too.
    """

    MIN_SAMPLES = 8
    WINDOW = 20.0       # seconds of history
    BEST = 0.3          # fraction of pings (lowest round trip) that are used
    MAX_DRIFT = 5e-4    # clocks never disagree by more than 500 ppm

    def __init__(self):
        self.samples: deque = deque()
        self.estimate = None    # (t_ref, offset, drift); swapped atomically
        self.rtt = None         # typical round trip of the trusted pings

    def add(self, t1: float, t2: float, t3: float, t4: float) -> None:
        rtt = (t4 - t1) - (t3 - t2)
        if rtt < 0:
            return
        mid = (t1 + t4) / 2
        self.samples.append((mid, ((t2 - t1) + (t3 - t4)) / 2, rtt))
        while mid - self.samples[0][0] > self.WINDOW:
            self.samples.popleft()
        if len(self.samples) < self.MIN_SAMPLES:
            return

        s = np.array(self.samples)
        keep = max(self.MIN_SAMPLES, int(len(s) * self.BEST))
        best = s[np.argsort(s[:, 2])[:keep]]
        t_ref = s[-1, 0]
        dt, off = best[:, 0] - t_ref, best[:, 1]
        drift = 0.0
        if np.ptp(dt) > 4.0:
            drift = float(np.clip(np.polyfit(dt, off, 1)[0], -self.MAX_DRIFT, self.MAX_DRIFT))
        offset = float(np.median(off - drift * dt))
        if self.estimate is not None:       # move a fifth of the way: steadier, still follows
            last_ref, last_offset, last_drift = self.estimate
            expected = last_offset + last_drift * (t_ref - last_ref)
            if abs(offset - expected) < 0.005:
                offset = expected + 0.2 * (offset - expected)
        self.estimate = (t_ref, offset, drift)
        self.rtt = float(np.median(best[:, 2]))

    def to_server(self, t_local: float):
        est = self.estimate
        if est is None:
            return None
        t_ref, offset, drift = est
        return t_local + offset + drift * (t_local - t_ref)


# ----------------------------------------------------------------- playback
class Settings:
    """Per-Mac knobs, changed live from the keyboard."""

    def __init__(self, offset_ms: float = 0.0, volume: float = 1.0, channel: str = "both"):
        self.offset_ms = offset_ms    # + plays this Mac later
        self.volume = volume
        self.channel = channel        # both | left | right


def stretch(seg: np.ndarray, frames: int) -> np.ndarray:
    """Linearly resample seg to `frames` frames (used for +-1 frame corrections)."""
    m = len(seg)
    x = np.arange(frames) * (m / frames)
    xp = np.arange(m)
    return np.stack([np.interp(x, xp, seg[:, c]) for c in range(seg.shape[1])], axis=1).astype(np.float32)


class Player:
    """Plays a timestamped stream so that every sample leaves the speaker at
    its play_at time on the host's clock.

    Each output callback knows when its first frame will reach the speaker.
    That gives the sample index that should be playing, which is compared with
    the one actually being read. Small errors (clock drift) are trimmed by
    adding or skipping single frames now and then; large ones (start-up, delay
    changes) by jumping straight to the right sample.
    """

    HARD_RESYNC = 0.02      # seconds off -> jump (while getting in sync)
    LOCKED_RESYNC = 0.04    # once in sync, a jump (which you hear) only beyond this
    GAIN = 0.5              # 1/s: how hard small errors are corrected
    MAX_SKEW = 0.002        # normally never run more than 0.2 % fast or slow
    FAST_SKEW = 0.006       # up to 0.6 % to win back a bigger slip without a jump
    SMOOTH = 0.05           # error filter, per callback
    # When each buffer reaches the ear is counted from the samples the output takes,
    # which is exact. macOS's own report of it is only used to line up at the start,
    # to notice real jumps, and to follow drift no bigger than a sound card's crystal.
    # Bluetooth outputs (AirPods, Beats) report times that wander and slide by several
    # percent while the headphones play perfectly steadily; chasing those reports made
    # the player slip and jump back every few seconds, which is what the static was.
    FOLLOW = 0.02           # share of the gap to the report closed per buffer
    MAX_DRIFT = 0.0005      # ...but never more than 0.05 % (crystals are within 0.01 %)
    TRUST = 0.02            # a report further than this from the sample count is ignored
    JUMP = 0.05             # a sudden step this big that lasts 3 buffers is a real jump

    def __init__(self, rate: int, channels: int, to_server, settings: Settings | None = None,
                 buffer_s: float = 4.0):
        self.rate = rate
        self.channels = channels
        self.to_server = to_server
        self.settings = settings or Settings()
        self.cap = int(rate * buffer_s)
        self.ring = np.zeros((self.cap, channels), np.float32)
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.n_first = self.n_end = None    # range of sample indices received
            self.anchor = None                  # (index, play_at) from the newest block
            self.read_n = None                  # next sample index to play
            self.acc = 0.0
            self.err = 0.0                      # filtered timing error, seconds (+ = early)
            self.starving = False
            self.synced = False
            self.locked = False                 # has been in sync since the last jump
            self.dropouts = 0
            self.resyncs = 0
            self.margins: deque = deque(maxlen=200)    # spare audio after each callback, seconds
            self.last_sound = None                      # when a block with actual sound last arrived
            self._dac = None
            self._dac_frames = 0
            self._gap = 0.0                     # last ordinary report minus the sample count
            self._dac_odd = 0                   # buffers in a row whose reported time made no sense

    def feed(self, n: int, play_at: float, pcm: np.ndarray) -> None:
        k = len(pcm)
        with self.lock:
            if self.n_end is None or n < self.n_end or n - self.n_end >= self.cap:
                self.n_first = self.n_end = n   # first block, or a new stream
                self.read_n = None
            elif n > self.n_end:
                self._write(self.n_end, np.zeros((n - self.n_end, self.channels), np.float32))
            self._write(n, pcm)
            self.n_end = n + k
            self.anchor = (n, play_at)
            if k and float(np.abs(pcm).max()) > 1e-4:
                self.last_sound = time.monotonic()

    def _write(self, n: int, pcm: np.ndarray) -> None:
        i, k = n % self.cap, len(pcm)
        first = min(k, self.cap - i)
        self.ring[i:i + first] = pcm[:first]
        self.ring[:k - first] = pcm[first:]

    def _read(self, start: int, need: int) -> np.ndarray:
        out = np.zeros((need, self.channels), np.float32)
        missing = start + need > self.n_end
        if missing and not self.starving:
            self.dropouts += 1          # audio wasn't here in time
        self.starving = missing
        if missing:
            # Clean silence: playing the bit that did arrive, then a hole, over
            # and over, is what crackle and "static" are made of.
            return out
        lo = max(start, self.n_first, self.n_end - self.cap)
        hi = start + need
        if hi > lo:
            i, m = lo % self.cap, hi - lo
            first = min(m, self.cap - i)
            out[lo - start:lo - start + first] = self.ring[i:i + first]
            out[lo - start + first:hi - start] = self.ring[:m - first]
        return out

    def _smooth_dac(self, dac: float, frames: int) -> float:
        """When this buffer reaches the speaker: the last one's time plus its length,
        nudged toward macOS's report only while that report is believable."""
        if self._dac is None:
            self._dac, self._dac_frames, self._gap, self._dac_odd = dac, frames, 0.0, 0
            return dac
        counted = self._dac + self._dac_frames / self.rate
        gap = dac - counted
        if abs(gap - self._gap) > self.JUMP:
            # A sudden step. AirPods now and then report one wild time (reacting to it
            # would be a click); a step that lasts means the output really jumped or restarted.
            self._dac_odd += 1
            if self._dac_odd >= 3 or abs(gap - self._gap) > 0.5:
                counted, self._gap, self._dac_odd = dac, 0.0, 0
        else:
            self._dac_odd = 0
            if abs(gap) < self.TRUST:
                most = self.MAX_DRIFT * frames / self.rate
                counted += min(most, max(-most, self.FOLLOW * gap))
            self._gap = gap
        self._dac, self._dac_frames = counted, frames
        return counted

    def new_output(self) -> None:
        """Playback moved to another device (AirPods in or out): line up with it afresh."""
        with self.lock:
            self._dac = None

    def render(self, frames: int, dac_local: float, out_channels: int = 2, shift: float = 0.0) -> np.ndarray:
        """Audio for `frames` frames whose first frame hits the speaker at dac_local.
        `shift`: how much the output's clock mapping (StreamClock) just moved; the time
        counted so far was in the old mapping, so it moves with it."""
        s = self.settings
        with self.lock:
            if shift and self._dac is not None:
                self._dac += shift
            server_t = self.to_server(self._smooth_dac(dac_local, frames))
            if server_t is None or self.anchor is None:
                self.read_n = None
                self.synced = False
                return np.zeros((frames, out_channels), np.float32)
            an, at = self.anchor
            target = an + (server_t - at - s.offset_ms / 1000) * self.rate
            limit = self.LOCKED_RESYNC if self.locked else self.HARD_RESYNC
            if self.read_n is None or abs(self.read_n - target) > limit * self.rate:
                if self.read_n is not None:
                    self.resyncs += 1
                self.read_n = int(round(target))
                self.acc = self.err = 0.0
                self.locked = False
                self.margins.clear()
            self.err += self.SMOOTH * ((self.read_n - target) / self.rate - self.err)
            most = self.FAST_SKEW if abs(self.err) > 0.008 else self.MAX_SKEW
            skew = min(most, max(-most, -self.GAIN * self.err))
            self.acc += skew * frames
            k = int(self.acc)
            self.acc -= k
            need = frames + k
            seg = self._read(self.read_n, need)
            if self.read_n + need > self.n_first:      # the stream has started
                self.margins.append((self.n_end - self.read_n - need) / self.rate)
            self.read_n += need
            # Shown as "in sync" within 3 ms; only shown as off beyond 10 ms, so the label
            # doesn't flicker on Wi-Fi's and Bluetooth's normal wobble (a few ms is inaudible).
            if self.starving or abs(self.err) > 0.010:
                self.synced = False
            elif abs(self.err) < 0.003:
                self.synced = self.locked = True
        if k:
            seg = stretch(seg, frames)
        return self._mix(seg, out_channels) * s.volume

    def _mix(self, seg: np.ndarray, out_channels: int) -> np.ndarray:
        mode = self.settings.channel
        if mode in ("left", "right") or out_channels == 1 or self.channels == 1:
            if mode == "left" or self.channels == 1:
                mono = seg[:, 0]
            elif mode == "right":
                mono = seg[:, 1]
            else:
                mono = seg.mean(axis=1)
            return np.repeat(mono[:, None], out_channels, axis=1)
        return seg[:, :out_channels]

    @property
    def in_sync(self) -> bool:
        return self.read_n is not None and self.synced

    def silent_for(self):
        """Seconds since any sound arrived (None before anything arrived)."""
        if self.n_end is None:
            return None
        return time.monotonic() - self.last_sound if self.last_sound else float("inf")

    def headroom(self):
        """How much audio was left over at the tightest recent moment, in
        seconds (negative: it ran out). This is what the delay has to cover."""
        margins = list(self.margins)
        return min(margins) if margins else None


# ------------------------------------------------------------------ capture
class Timeline:
    """Gives captured blocks a smooth timestamp: sample n is captured at
    t0 + n / rate. t0 slowly follows the real capture times, so callback
    jitter never reaches the players but capture-clock drift does."""

    def __init__(self, rate: int):
        self.rate = rate
        self.n = 0
        self.t0 = None
        self._jumps = 0

    def stamp(self, frames: int, captured_at: float):
        n = self.n
        if self.t0 is None:
            self.t0 = captured_at
        diff = captured_at - (self.t0 + n / self.rate)
        if abs(diff) > 0.02:
            self._jumps += 1
            if self._jumps >= 5:            # sustained: the device stalled or skipped
                self.t0 += diff
                self._jumps = 0
        else:
            self._jumps = 0
            self.t0 += min(5e-5, max(-5e-5, diff * 0.005))
        self.n += frames
        return n, self.t0 + n / self.rate


def make_click(rate: int, freq: float) -> np.ndarray:
    t = np.arange(int(0.015 * rate)) / rate
    return (np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.003) * 0.7).astype(np.float32)


class ClickSource:
    """A metronome (accent every 4th beat) used instead of BlackHole, so the
    sync can be checked by ear: one crisp click = in sync, a 'flam' = not."""

    def __init__(self, rate: int, channels: int, on_block, now=time.monotonic):
        self.rate, self.channels, self.on_block, self.now = rate, channels, on_block, now
        self.hi, self.lo = make_click(rate, 2000), make_click(rate, 1000)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self._stop.set()

    def _run(self):
        block, period = self.rate * BLOCK_MS // 1000, self.rate // 2
        start, n = self.now() + 0.05, 0
        while not self._stop.is_set():
            wait = start + (n + block) / self.rate - self.now()
            if wait > 0:
                time.sleep(wait)
            idx = np.arange(n, n + block)
            phase, beat = idx % period, idx // period
            hit = phase < len(self.hi)
            pcm = np.zeros((block, self.channels), np.float32)
            pcm[hit] = np.where(beat[hit] % 4 == 0, self.hi[phase[hit]], self.lo[phase[hit]])[:, None]
            self.on_block(pcm, start + n / self.rate)
            n += block


# ------------------------------------------------------------- audio devices
class SetupError(Exception):
    """A problem the person can fix; `code` tells the app which help to show."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def import_sounddevice():
    try:
        import sounddevice
    except (ImportError, OSError) as e:
        sys.exit(f"Can't load the audio library ({e}). Run ./setup.sh first.")
    return sounddevice


def refresh_portaudio(sd):
    """PortAudio only knows the devices that existed when it started (AirPods
    connected later are invisible). Restart it; no streams may be open."""
    if hasattr(sd, "_terminate"):
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass


def find_device(sd, query: str, kind: str) -> int:
    devices = sd.query_devices()
    usable = [i for i, d in enumerate(devices) if d[f"max_{kind}_channels"] > 0]
    if query.isdigit() and int(query) in usable:
        return int(query)
    q = query.lower()
    for match in (lambda name: name == q, lambda name: q in name):
        hits = [i for i in usable if match(devices[i]["name"].lower())]
        if hits:
            return hits[0]
    names = ", ".join(f'"{devices[i]["name"]}"' for i in usable)
    if kind == "input" and q == "blackhole":
        raise SetupError("no_blackhole", "BlackHole isn't installed yet. It's the free audio driver that "
                         "lets SyncAudio hear what this Mac plays.")
    raise SetupError("no_device", f'No {kind} device matching "{query}". Available: {names}')


def is_combined(name: str) -> bool:
    """Multi-Output / Aggregate devices people make for BlackHole often contain
    BlackHole itself: playing into one would feed the share back into itself."""
    name = name.lower()
    return "blackhole" in name or "multi-output" in name or "aggregate" in name


def pick_output(sd, query: str | None) -> int:
    """The named device, else the current default output (unless that is
    BlackHole), else the built-in speakers."""
    if query:
        return find_device(sd, query, "output")
    devices = sd.query_devices()
    default = sd.default.device[1]
    if default is not None and 0 <= default < len(devices) and not is_combined(devices[default]["name"]):
        return default
    outputs = [i for i, d in enumerate(devices)
               if d["max_output_channels"] > 0 and not is_combined(d["name"])]
    speakers = [i for i in outputs if "speaker" in devices[i]["name"].lower()]
    if speakers or outputs:
        return (speakers or outputs)[0]
    raise SetupError("no_device", "No speakers found.")


class StreamClock:
    """Turns PortAudio's timestamps into time.monotonic() time.

    The audio callbacks are Python, so they run when Python gets round to them
    (another thread may hold the GIL), sometimes many milliseconds after macOS
    asked for audio. Timing a buffer by "now" inside the callback wanders with
    that delay, and the players chase the wander: "Getting in sync", then a jump
    you hear. PortAudio's own timestamps are taken on time, so use those. The
    offset between the two clocks is the smallest difference seen: the callback
    that ran soonest. It creeps up very slowly, in case the clocks run at
    slightly different rates (on a Mac they are the same clock)."""

    CREEP = 2e-5        # seconds per second

    def __init__(self, now=time.monotonic):
        self.now = now
        self._offset = self._last = None

    def offset(self, stream_now: float) -> float:
        diff = self.now() - stream_now
        if self._offset is None or diff < self._offset:
            self._offset = diff
        else:
            self._offset += max(0.0, stream_now - self._last) * self.CREEP
        self._last = stream_now
        return self._offset


class SoundDeviceOutput:
    def __init__(self, sd, device: int, rate: int, render, now=time.monotonic):
        info = sd.query_devices(device, "output")
        self.name = info["name"]
        self.channels = min(2, info["max_output_channels"])
        self.underflows = 0         # macOS ran out of audio to play: a click
        clock = StreamClock(now)
        good = []                   # the last sensible lead time
        mapped = []                 # the clock offset used for the last buffer
        failed = []

        def callback(outdata, frames, t, status):
            if status and status.output_underflow:
                self.underflows += 1
            try:
                offset = clock.offset(t.currentTime)
                lead = t.outputBufferDacTime - t.currentTime
                if 0.0 <= lead < 1.0:
                    good[:] = [lead]
                else:                              # timestamps unavailable: don't jump
                    lead = good[0] if good else self.stream.latency
                shift = offset - mapped[0] if mapped else 0.0
                mapped[:] = [offset]
                outdata[:] = render(frames, t.currentTime + offset + lead, self.channels, shift)
            except Exception as e:  # never let an exception kill the stream
                outdata.fill(0)
                if not failed:
                    failed.append(e)
                    print(f"\nplayback error: {e!r}", file=sys.stderr)

        self.stream = sd.OutputStream(samplerate=rate, device=device, channels=self.channels,
                                      dtype="float32", latency=OUTPUT_LATENCY, callback=callback)
        self.stream.start()
        self.latency = self.stream.latency

    def close(self):
        self.stream.stop()
        self.stream.close()


class SoundDeviceInput:
    def __init__(self, sd, device: int, rate: int, channels: int, on_block, now=time.monotonic):
        self.name = sd.query_devices(device, "input")["name"]
        self.overflows = 0          # capture fell behind and lost audio: a click for everyone
        clock = StreamClock(now)
        failed = []

        def callback(indata, frames, t, status):
            if status and status.input_overflow:
                self.overflows += 1
            try:
                offset = clock.offset(t.currentTime)
                back = t.currentTime - t.inputBufferAdcTime
                if not 0.0 <= back < 1.0:
                    back = frames / rate
                on_block(indata.copy(), t.currentTime + offset - back)
            except Exception as e:  # keep capturing whatever happens
                if not failed:
                    failed.append(e)
                    print(f"\ncapture error: {e!r}", file=sys.stderr)

        self.stream = sd.InputStream(samplerate=rate, device=device, channels=channels,
                                     dtype="float32", blocksize=rate * BLOCK_MS // 1000,
                                     latency=INPUT_LATENCY, callback=callback)

    def start(self):
        self.stream.start()

    def close(self):
        self.stream.stop()
        self.stream.close()


# --------------------------------------------------------------------- host
class _Peer:
    QUEUE = 300     # blocks waiting to be sent (3 s at 10 ms blocks)
    SNDBUF = 64 * 1024      # the kernel's share of the backlog: ~1/3 s of audio

    def __init__(self, sock: socket.socket, name: str, ip: str):
        self.sock, self.name, self.ip = sock, name, ip
        self.q: queue.Queue = queue.Queue(maxsize=self.QUEUE)
        self.stats: dict = {}
        self.version = None
        self.joined = time.monotonic()
        self.alive = True
        self.late = 0           # blocks skipped because the network couldn't deliver them in time
        self.codec = "pcm"      # "opus" when both Macs can: a tenth of the Wi-Fi traffic

    def push(self, data: bytes, play_at: float = math.inf):
        """Queue a frame; audio carries the time it must be playing by."""
        while True:
            try:
                self.q.put_nowait((play_at, data))
                return
            except queue.Full:
                try:                # Wi-Fi fell far behind: the oldest audio is useless now
                    if self.q.get_nowait()[0] != math.inf:
                        self.late += 1
                except queue.Empty:
                    pass

    def take(self, now: float, timeout: float = 0.5) -> bytes:
        """Everything queued, minus audio that would reach the speaker too late.
        Sending that would only build a backlog, and the listener would fall
        further behind with every Wi-Fi hiccup until it gave up."""
        items = [self.q.get(timeout=timeout)]
        while True:
            try:
                items.append(self.q.get_nowait())
            except queue.Empty:
                break
        keep = [data for play_at, data in items if play_at > now]
        self.late += len(items) - len(keep)
        return b"".join(keep)


class Host:
    def __init__(self, rate: int, channels: int, delay: float, port: int = PORT,
                 name: str = "host", log=print, now=time.monotonic):
        self.rate, self.channels, self.delay = rate, channels, delay
        self.port, self.name, self.log, self.now = port, name, log, now
        self.timeline = Timeline(rate)
        self.local: Player | None = None
        self.peers: set = set()
        self.lock = threading.Lock()
        self.peak = 0.0
        self.last_sound = None
        self.group_volume = 1.0     # everyone's volume; any Mac may change it
        self.group_volume_by = None
        self._gain = 1.0            # what was applied to the last block
        self._encoder = None        # Opus, made when the first Mac that can decode it joins
        self.running = False

    def start(self):
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.tcp.bind(("", self.port))
            self.tcp.listen()
            self.udp.bind(("", self.port))
        except OSError:
            self.tcp.close()
            self.udp.close()
            raise
        # Blocking calls don't notice a socket being closed under them, so the
        # loops wake up regularly to check whether they should stop.
        self.tcp.settimeout(0.5)
        self.udp.settimeout(0.5)
        self.running = True
        self._threads = [threading.Thread(target=self._accept_loop, daemon=True),
                         threading.Thread(target=self._udp_loop, daemon=True)]
        for t in self._threads:
            t.start()

    def stop(self):
        """Stop serving; the port is free again once this returns."""
        self.running = False
        for t in getattr(self, "_threads", []):
            t.join(timeout=2)
        for s in (self.tcp, self.udp):
            try:
                s.close()
            except OSError:
                pass
        with self.lock:
            for p in self.peers:
                p.alive = False

    def _announce(self, **changes):
        data = frame(UPDATE, json.dumps(changes).encode())
        with self.lock:
            peers = list(self.peers)
        for p in peers:
            p.push(data)

    def set_delay(self, delay: float):
        """Change the capture-to-speaker delay for everyone, live."""
        self.delay = delay
        self._announce(delay=delay)

    def set_group_volume(self, volume: float, by: str | None = None):
        """Everyone's volume (0..1). Applied here, before the audio is sent."""
        self.group_volume = min(1.0, max(0.0, float(volume)))
        self.group_volume_by = by or self.name
        self._announce(group_volume=self.group_volume, by=self.group_volume_by)

    def on_block(self, pcm: np.ndarray, captured_at: float):
        """Called by the capture device for every block (audio thread)."""
        n, t = self.timeline.stamp(len(pcm), captured_at)
        play_at = t + self.delay
        peak = float(np.abs(pcm).max(initial=0.0))
        self.peak = max(self.peak, peak)
        if peak > 1e-4:
            self.last_sound = time.monotonic()
        target = self.group_volume
        if target != self._gain:        # glide to the new volume over one block: no click
            pcm = pcm * np.linspace(self._gain, target, len(pcm), dtype=np.float32)[:, None]
            self._gain = target
        elif target != 1.0:
            pcm = pcm * np.float32(target)
        if self.local:
            self.local.feed(n, play_at, pcm)
        with self.lock:
            peers = list(self.peers)
        if peers:
            data = {}
            if any(p.codec == "pcm" for p in peers):
                pcm16 = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
                data["pcm"] = frame(AUDIO, AUDIO_HDR.pack(n, play_at) + pcm16)
            if any(p.codec == "opus" for p in peers):
                if self._encoder is None:
                    self._encoder = codec.Encoder(self.channels)
                packet = self._encoder.encode(pcm)
                if packet:          # (a block of an odd size is skipped: the listener hears a gap)
                    data["opus"] = frame(OPUS, AUDIO_HDR.pack(n, play_at) + packet)
            for p in peers:
                if p.codec in data:
                    p.push(data[p.codec], play_at)

    def take_level(self) -> float:
        """Loudest sample since the last call (0..1)."""
        level, self.peak = self.peak, 0.0
        return level

    def listeners(self) -> list:
        with self.lock:
            peers = sorted((p for p in self.peers if not getattr(p, "probe", False)), key=lambda p: p.name.lower())
        return [{"state": "waiting", **p.stats, "name": p.name, "ip": p.ip, "version": p.version,
                 "connected_s": time.monotonic() - p.joined, "late": p.late, "codec": p.codec}
                for p in peers]

    def _udp_loop(self):
        while self.running:
            try:
                data, addr = self.udp.recvfrom(512)
                t2 = self.now()
                if len(data) == PING.size and data[:1] == b"P":
                    _, seq, t1 = PING.unpack(data)
                    self.udp.sendto(PONG.pack(b"Q", seq, t1, t2, self.now()), addr)
                elif data == DISCOVER:
                    self.udp.sendto(b"d" + json.dumps({"name": self.name, "port": self.port}).encode(), addr)
            except OSError:
                if not self.running:
                    return

    def _accept_loop(self):
        while self.running:
            try:
                sock, addr = self.tcp.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._serve, args=(sock, addr[0]), daemon=True).start()

    def _serve(self, sock: socket.socket, ip: str):
        peer = None
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _Peer.SNDBUF)
            sock.settimeout(5)
            kind, payload = recv_frame(sock)
            hello = json.loads(payload)
            if kind != HELLO or hello.get("magic") != MAGIC:
                return
            opus = "opus" in (hello.get("codecs") or []) and codec.available(self.rate, self.channels)
            sock.sendall(frame(HEADER, json.dumps({
                "rate": self.rate, "channels": self.channels, "delay": self.delay,
                "group_volume": self.group_volume, "name": self.name, "version": APP_VERSION,
                "codec": "opus" if opus else "pcm"}).encode()))
            peer = _Peer(sock, str(hello.get("name") or ip)[:40], ip)
            peer.probe = bool(hello.get("probe"))
            peer.version = hello.get("version")
            peer.codec = "opus" if opus else "pcm"
            with self.lock:
                self.peers.add(peer)
            if not peer.probe:
                self.log(f"+ {peer.name} joined ({ip})")
            threading.Thread(target=self._read_stats, args=(peer,), daemon=True).start()
            while self.running and peer.alive:
                try:
                    data = peer.take(self.now())
                except queue.Empty:
                    continue
                if data:
                    sock.sendall(data)
        except (OSError, ValueError, ConnectionError):
            pass
        finally:
            if peer:
                peer.alive = False
                with self.lock:
                    self.peers.discard(peer)
                if self.running and not peer.probe:
                    self.log(f"- {peer.name} left")
            sock.close()

    def _read_stats(self, peer: _Peer):
        try:
            while peer.alive:
                kind, payload = recv_frame(peer.sock)
                if kind == STATS:
                    peer.stats = json.loads(payload)
                elif kind == CONTROL:
                    request = json.loads(payload)
                    if "group_volume" in request:
                        self.set_group_volume(request["group_volume"], by=peer.name)
                        self.log(f"{peer.name} set everyone's volume to {self.group_volume:.0%}")
        except (OSError, ValueError, ConnectionError):
            peer.alive = False


class AutoDelay:
    """Picks the host's delay: as short as possible, but long enough that
    every listener gets each block before it is due. Raises at once when a
    listener runs short; lowers only after a long calm stretch."""

    START, MIN, MAX = 0.25, 0.12, 1.5   # 100 ms output buffer + capture buffer + Wi-Fi
    MARGIN = 0.05       # spare audio every speaker should keep (Wi-Fi and thread hiccups)
    SETTLE = 5.0        # seconds listeners need to measure a change
    CALM = 120.0        # seconds of plenty of headroom before lowering
    NO_LOWER = 600.0    # after a raise, don't lower for this long
    STEP = 0.3          # most a single raise adds: one long Wi-Fi stall isn't the new normal

    def __init__(self, delay: float = START):
        self.delay = delay
        self.changed_at = self.raised_at = -math.inf
        self.calm_since = None
        self.drops: dict = {}

    def update(self, now: float, listeners: dict):
        """listeners: {key: stats from Client.stats()}. Returns a new delay or None."""
        reports = {k: s for k, s in listeners.items() if s.get("headroom_ms") is not None}
        dropped = any(s.get("dropouts", 0) > self.drops.get(k, s.get("dropouts", 0))
                      for k, s in reports.items())
        self.drops = {k: s.get("dropouts", 0) for k, s in reports.items()}
        if not reports or now - self.changed_at < self.SETTLE:
            return None
        worst = min(s["headroom_ms"] for s in reports.values()) / 1000
        if dropped or worst < self.MARGIN / 2:
            new = self.delay + min(self.STEP, max(0.05, self.MARGIN - worst))
            self.raised_at = now
        elif worst > 4 * self.MARGIN and now - self.raised_at > self.NO_LOWER:
            if self.calm_since is None:
                self.calm_since = now
            if now - self.calm_since < self.CALM:
                return None
            new = self.delay - min(0.05, worst - 2 * self.MARGIN)
        else:
            self.calm_since = None
            return None
        new = min(self.MAX, max(self.MIN, math.ceil(new * 100 - 1e-9) / 100))
        if abs(new - self.delay) < 0.005:
            return None
        self.delay, self.changed_at, self.calm_since = new, now, None
        return new


# ------------------------------------------------------------------- client
def broadcast_targets() -> set:
    """Where to send discovery: broadcast addresses, plus every address on
    this Mac's /24 subnet for routers that swallow broadcasts."""
    targets = {"255.255.255.255"}
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True, timeout=2).stdout
        targets.update(re.findall(r"broadcast (\d+\.\d+\.\d+\.\d+)", out))
    except (OSError, subprocess.SubprocessError):
        pass
    ip = lan_ip()
    if not ip.startswith("127."):
        prefix = ip.rsplit(".", 1)[0]
        targets.update(f"{prefix}.{i}" for i in range(1, 255))
        targets.discard(ip)
    return targets


def discover_all(port: int = PORT, timeout: float = 1.0, targets=None, errors: list | None = None) -> list:
    """Every host answering on the local network, as [(ip, name)]. Send
    failures are appended to `errors` (errno values) when given."""
    found = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        for target in (broadcast_targets() if targets is None else targets):
            try:
                s.sendto(DISCOVER, (target, port))
            except OSError as e:
                if errors is not None:
                    errors.append(e.errno)
        end = time.monotonic() + timeout
        while (left := end - time.monotonic()) > 0:
            s.settimeout(left)
            try:
                data, addr = s.recvfrom(512)
            except OSError:
                break
            if data[:1] == b"d":
                try:
                    found[addr[0]] = str(json.loads(data[1:]).get("name", addr[0]))
                except ValueError:
                    pass
    return sorted(found.items(), key=lambda item: item[1].lower())


def discover(port: int = PORT, timeout: float = 1.5, address: str | None = None):
    """The first host found, as (ip, name), or None."""
    hosts = discover_all(port, timeout, [address] if address else None)
    return hosts[0] if hosts else None


def explain(e: BaseException) -> dict:
    """Turn a connection error into something a person can act on."""
    code = getattr(e, "errno", None)
    if isinstance(e, socket.timeout) or code == errno.ETIMEDOUT:
        return {"code": "timeout", "message": "No answer from that address. Either this Wi-Fi keeps devices "
                "apart (common on school, office and hotel networks) or the firewall on that Mac is blocking "
                "SyncAudio."}
    if code == errno.ECONNREFUSED:
        return {"code": "refused", "message": "That Mac answered, but it isn't sharing. Open SyncAudio there "
                "and click Start sharing."}
    if code == errno.EHOSTUNREACH:
        return {"code": "local_network", "message": "macOS is blocking SyncAudio from the local network. "
                "Turn on SyncAudio in System Settings → Privacy & Security → Local Network, then reopen it."}
    if code == getattr(errno, "EHOSTDOWN", -1):
        return {"code": "no_host", "message": "Nothing answers at that address. Check it's typed right and "
                "that both Macs are on the same Wi-Fi."}
    if code == errno.ENETUNREACH:
        return {"code": "no_network", "message": "This Mac isn't connected to a network."}
    return {"code": "other", "message": f"Couldn't connect ({e})."}


def probe(address: str, port: int = PORT, seconds: float = 2.0) -> dict:
    """Check a sharing Mac step by step: answers discovery, accepts the
    connection, streams audio, answers clock pings."""
    result = {"address": address, "port": port, "ok": False,
              "discovery": bool(discover_all(port, 1.0, [address]))}
    try:
        sock = socket.create_connection((address, port), timeout=3)
    except OSError as e:
        return {**result, "step": "connect", "problem": explain(e)}
    try:
        sock.settimeout(3)
        sock.sendall(frame(HELLO, json.dumps({"magic": MAGIC, "name": "connection test", "probe": True}).encode()))
        kind, payload = recv_frame(sock)
        header = json.loads(payload)
        blocks, size, end = 0, 0, time.monotonic() + seconds
        while time.monotonic() < end:
            kind, payload = recv_frame(sock)
            if kind == AUDIO:
                blocks, size = blocks + 1, size + len(payload)
    except (OSError, ValueError, ConnectionError) as e:
        return {**result, "step": "stream", "problem": explain(e)}
    finally:
        sock.close()
    rtts = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(0.5)
        for seq in range(10):
            t1 = time.monotonic()
            try:
                s.sendto(PING.pack(b"P", seq, t1), (address, port))
                if PONG.unpack(s.recv(64))[1] == seq:
                    rtts.append(time.monotonic() - t1)
            except (OSError, struct.error):
                pass
    return {**result, "ok": blocks > 0 and bool(rtts), "step": "done", "host_name": header.get("name"),
            "audio_blocks_per_s": round(blocks / seconds), "audio_kbps": round(size * 8 / seconds / 1000),
            "wifi_ms": round(float(np.median(rtts)) * 1000, 1) if rtts else None, "pings_answered": len(rtts)}


def player_state(p: Player | None) -> str:
    if p is None or p.read_n is None:
        return "waiting"
    if p.starving:
        return "no_audio"
    return "in_sync" if p.in_sync else "syncing"


class Client:
    def __init__(self, host: str | None, open_output, port: int = PORT, name: str = "friend",
                 settings: Settings | None = None, log=print, now=time.monotonic,
                 discover_address: str | None = None):
        self.host, self.open_output, self.port, self.name = host, open_output, port, name
        self.settings = settings or Settings()
        self.log, self.now, self.discover_address = log, now, discover_address
        self.clock = ClockSync()
        self.player: Player | None = None
        self.output = None
        self.host_name = None
        self.host_version = None
        self.codec = None               # how the audio arrives: "opus" or "pcm"
        self.delay = None
        self.group_volume = None
        self.group_volume_by = None
        self._sock = None
        self._send_lock = threading.Lock()
        self._output_lock = threading.Lock()
        self.phase = "searching"        # searching | connecting | connected | retrying
        self.error = None
        self.problem = None             # why we can't connect, for people (see explain)
        self.running = False

    @property
    def state(self) -> str:
        return {"searching": "looking for the host", "connecting": "connecting",
                "connected": "playing", "retrying": f"can't reach {self.host}:{self.port}, retrying"}[self.phase]

    def to_server(self, t: float):
        return self.clock.to_server(t)

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False
        with self._output_lock:
            if self.output:
                self.output.close()
                self.output = None

    def reopen_output(self, before_open=None):
        """Move playback to whatever device open_output picks now."""
        with self._output_lock:
            if self.output:
                self.output.close()
                self.output = None
            if before_open:
                before_open()
            if self.player and self.running:
                self.player.new_output()
                self.output = self.open_output(self.player.rate, self.player.render)

    def _run(self):
        while self.running:
            ip = self.host
            if not ip:
                self.phase = "searching"
                found = discover(self.port, address=self.discover_address)
                if not found:
                    continue
                ip, self.host_name = found
            try:
                self.phase = "connecting"
                self._session(ip)
            except (OSError, ValueError, KeyError, ConnectionError) as e:
                if self.running and self.phase == "connected":
                    self.log(f"lost the host ({e}); reconnecting...")
                elif isinstance(e, OSError):
                    self.problem = explain(e)
            except Exception as e:  # e.g. the speaker can't be opened
                self.error = str(e)
                self.log(f"error: {e!r}")
                time.sleep(2)
            if self.running:
                self.phase = "retrying" if self.host else "searching"
                time.sleep(1)

    def _session(self, ip: str):
        sock = socket.create_connection((ip, self.port), timeout=3)
        stop = threading.Event()
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.sendall(frame(HELLO, json.dumps({"magic": MAGIC, "name": self.name, "version": APP_VERSION,
                                                  "codecs": ["opus"] if codec.available() else []}).encode()))
            kind, payload = recv_frame(sock)
            if kind != HEADER:
                raise ConnectionError("not a syncaudio host")
            hdr = json.loads(payload)
            self.host_version = hdr.get("version")      # None: a version from before versions were sent
            rate, channels = int(hdr["rate"]), int(hdr["channels"])
            self.host_name = hdr.get("name", ip)
            self.delay = float(hdr.get("delay", 0))
            self.group_volume = float(hdr.get("group_volume", 1.0))
            self.codec = hdr.get("codec", "pcm")
            decoder = codec.Decoder(channels) if self.codec == "opus" else None
            self.clock = ClockSync()
            with self._output_lock:
                if self.player is None or (self.player.rate, self.player.channels) != (rate, channels):
                    if self.output:
                        self.output.close()
                        self.output = None
                    self.player = Player(rate, channels, self.to_server, self.settings)
                    self.output = self.open_output(rate, self.player.render)
                else:
                    self.player.reset()
            self.error = self.problem = None
            self._sock = sock
            threading.Thread(target=self._ping_loop, args=(ip, stop), daemon=True).start()
            self.phase = "connected"
            self.log(f"connected to {self.host_name} ({ip})")
            last_stats = self.now()
            while self.running:
                kind, payload = recv_frame(sock)
                if kind == AUDIO:
                    n, play_at = AUDIO_HDR.unpack_from(payload)
                    pcm = np.frombuffer(payload, "<i2", offset=AUDIO_HDR.size)
                    self.player.feed(n, play_at, pcm.reshape(-1, channels).astype(np.float32) / 32768)
                elif kind == OPUS and decoder:
                    n, play_at = AUDIO_HDR.unpack_from(payload)
                    pcm = decoder.decode(payload[AUDIO_HDR.size:])
                    # Opus hands each block back 2.5 ms late: place it where it really belongs.
                    self.player.feed(n - codec.DELAY, play_at - codec.DELAY / rate, pcm)
                elif kind == UPDATE:
                    update = json.loads(payload)
                    self.delay = float(update.get("delay", self.delay))
                    if "group_volume" in update:
                        self.group_volume = float(update["group_volume"])
                        self.group_volume_by = update.get("by")
                if self.now() - last_stats >= 1.0:
                    last_stats = self.now()
                    self._send(STATS, self.stats())
        finally:
            stop.set()
            self._sock = None
            sock.close()

    def _send(self, kind: bytes, message: dict):
        sock = self._sock
        if sock:
            with self._send_lock:
                sock.sendall(frame(kind, json.dumps(message).encode()))

    def set_group_volume(self, volume: float):
        """Ask the host to change everyone's volume."""
        self.group_volume = min(1.0, max(0.0, float(volume)))
        self.group_volume_by = self.name
        try:
            self._send(CONTROL, {"group_volume": self.group_volume})
        except OSError:
            pass

    def _ping_loop(self, ip: str, stop: threading.Event):
        started = self.now()
        seq = 0
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            while not stop.is_set():
                seq += 1
                try:
                    s.sendto(PING.pack(b"P", seq, self.now()), (ip, self.port))
                    s.settimeout(0.25)
                    while True:
                        data = s.recv(64)
                        t4 = self.now()
                        if len(data) == PONG.size and data[:1] == b"Q":
                            _, rseq, t1, t2, t3 = PONG.unpack(data)
                            if rseq == seq:
                                self.clock.add(t1, t2, t3, t4)
                                break
                except OSError:
                    pass
                stop.wait(0.03 if self.now() - started < 2 else 0.2)

    def headroom(self):
        return self.player.headroom() if self.player else None

    def stats(self) -> dict:
        p = self.player
        room = self.headroom()
        return {
            "state": player_state(p),
            "synced": bool(p and p.in_sync),
            "err_ms": round(p.err * 1000, 2) if p else None,
            "wifi_ms": round(self.clock.rtt * 1000, 1) if self.clock.rtt is not None else None,
            "headroom_ms": round(room * 1000) if room is not None else None,
            "dropouts": p.dropouts if p else 0,
            "resyncs": p.resyncs if p else 0,
            "underflows": getattr(self.output, "underflows", 0),
            "glitches": glitches(p, self.output),
            "report_gap_ms": round(p._gap * 1000, 1) if p else None,   # macOS's timing report vs the samples
            "codec": self.codec,
        }

    def status(self) -> str:
        if self.phase != "connected" or not self.player:
            return self.state
        p, parts = self.player, [f"{self.host_name}"]
        if self.clock.estimate is None:
            parts.append("syncing clocks")
        else:
            parts.append(sync_word(p))
            parts.append(f"wifi {self.clock.rtt * 1000:.1f} ms")
        room = self.headroom()
        if room is not None:
            parts.append(f"headroom {room * 1000:.0f} ms" + (" ⚠ audio arriving late" if room < 0.01 else ""))
        return " · ".join(parts)


# ----------------------------------------------------------------- sessions
class SystemOutput:
    """Points the Mac's sound output at BlackHole while hosting, and back after."""

    def __init__(self, log, tool: str | None = None):
        self.tool = tool or shutil.which("SwitchAudioSource")
        self.log = log
        self.previous = None

    def _run(self, *args) -> str:
        try:
            return subprocess.run([self.tool, *args, "-t", "output"], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError) as e:
            self.log(f"! couldn't change the sound output: {e}")
            return ""

    def current(self) -> str:
        return self._run("-c") if self.tool else ""

    def switch_to(self, name: str, fallback: str | None):
        if not self.tool:
            self.log(f'! Set this Mac\'s sound output to "{name}" (Control Center → Sound) '
                     "so apps play into SyncAudio.")
            return
        current = self.current()
        self.previous = fallback if is_combined(current) else current
        if current != name:
            self._run("-s", name)
            self.log(f'Sound output switched to "{name}" (restored when you stop).')

    def set(self, name: str):
        if self.tool and name:
            self._run("-s", name)

    def restore(self):
        if self.tool and self.previous:
            self._run("-s", self.previous)
            self.previous = None


def glitches(p: Player | None, output=None) -> int:
    """Every audible hiccup: audio arriving late, sync jumps, macOS running short."""
    return (p.dropouts + p.resyncs if p else 0) + getattr(output, "underflows", 0)


def player_info(p: Player | None, output=None) -> dict:
    return {"state": player_state(p), "err_ms": round(p.err * 1000, 2) if p else None,
            "dropouts": p.dropouts if p else 0, "resyncs": p.resyncs if p else 0,
            "underflows": getattr(output, "underflows", 0), "glitches": glitches(p, output),
            "report_gap_ms": round(p._gap * 1000, 1) if p else None}


class HostSession:
    """Everything hosting needs: capture, playback on this Mac, the network
    side, switching the sound output to BlackHole, and the automatic delay."""

    def __init__(self, sd, name: str, delay_ms: float | None = None, click: bool = False,
                 play_local: bool = True, input_query: str = "BlackHole", output_query: str | None = None,
                 settings: Settings | None = None, port: int = PORT, switch_output: bool = True,
                 switch_tool: str | None = None, log=print):
        self.sd, self.name, self.click, self.play_local = sd, name, click, play_local
        self.input_query, self.output_query, self.port = input_query, output_query, port
        self.settings = settings or Settings()
        self.switch_output, self.switch_tool, self.log = switch_output, switch_tool, log
        self.auto = AutoDelay() if delay_ms is None else None
        self.fixed_delay = None if delay_ms is None else delay_ms / 1000
        self.host = self.output = self.source = self.system = self.advertiser = None
        self._stop = threading.Event()

    @property
    def delay(self) -> float:
        return self.host.delay if self.host else (self.auto.delay if self.auto else self.fixed_delay)

    def start(self):
        sd = self.sd
        out_dev = pick_output(sd, self.output_query) if self.play_local else None
        if self.click:
            rate, channels, in_dev = 48000, 2, None
        else:
            in_dev = find_device(sd, self.input_query, "input")
            info = sd.query_devices(in_dev, "input")
            rate, channels = int(info["default_samplerate"]), min(2, info["max_input_channels"])
        self.host = Host(rate, channels, self.delay, self.port, self.name, log=self.log)
        try:
            self.host.start()
        except OSError as e:
            self.host = None
            raise SetupError("port_busy", f"Can't share on port {self.port} ({e}). "
                                          "Is SyncAudio already sharing on this Mac?")
        try:
            if out_dev is not None:
                self.host.local = Player(rate, channels, lambda t: t, self.settings)
                self.output = SoundDeviceOutput(sd, out_dev, rate, self.host.local.render)
            self.system = SystemOutput(self.log, self.switch_tool)
            if self.click:
                self.source = ClickSource(rate, channels, self.host.on_block)
            else:
                self.source = SoundDeviceInput(sd, in_dev, rate, channels, self.host.on_block)
                if self.switch_output:
                    self.system.switch_to(self.source.name, self.output.name if self.output else None)
            self.source.start()
        except sd.PortAudioError as e:
            self.stop()
            raise SetupError("audio", f"Couldn't open the audio device ({e}).")
        except BaseException:
            self.stop()
            raise
        self.started = time.monotonic()
        # Never over the AirDrop link while sharing: it makes Wi-Fi hop channels and stall (dropouts).
        self.advertiser = discovery.Advertiser(self.name, self.port, lan_ip(), p2p_seconds=0).start()
        threading.Thread(target=self._tick_loop, daemon=True).start()
        return self

    def _tick_loop(self):
        ticks = 0
        while not self._stop.wait(1.0):
            ticks += 1
            if self.auto and self.host:
                now = time.monotonic()
                ready = {l["ip"] + l["name"]: l for l in self.host.listeners() if l["connected_s"] > 1.5}
                local = self.host.local
                if local and local.headroom() is not None:     # this Mac's own speaker or AirPods
                    ready["this Mac"] = {"headroom_ms": local.headroom() * 1000, "dropouts": local.dropouts}
                new = self.auto.update(now, ready)
                if new is not None:
                    self.host.set_delay(new)
                    self.log(f"delay set to {new * 1000:.0f} ms to keep everyone fed")
            if ticks % 2 == 0:
                try:
                    self._follow_output()
                except Exception as e:
                    self.log(f"! couldn't move to the new speaker: {e}")

    def _follow_output(self):
        """AirPods connecting (or picking another speaker) moves the Mac's sound
        output away from BlackHole. Play this Mac's share there instead, and
        point the output back at BlackHole so everyone keeps hearing the apps."""
        if not (self.system and self.system.tool) or self._stop.is_set():
            return
        current = self.system.current()
        changed, self._seen_output = current != getattr(self, "_seen_output", None), current
        if not current or current.lower().startswith("blackhole") or not changed:
            return
        if self.output and not self.output_query and current != self.output.name and not is_combined(current):
            self._reopen_audio(current)
            self.log(f'This Mac now plays on "{current}".')
        if not self.click and self.switch_output and isinstance(self.source, SoundDeviceInput):
            if not is_combined(current):
                self.system.previous = current
            self.system.set(self.source.name)
            self.log(f'Sound output was "{current}"; put back on "{self.source.name}" so everyone hears it.')

    def _reopen_audio(self, output_name: str):
        sd, h = self.sd, self.host
        restart_input = isinstance(self.source, SoundDeviceInput)
        for thing in ([self.source] if restart_input else []) + [self.output]:
            try:
                thing.close()
            except Exception:
                pass
        refresh_portaudio(sd)
        try:
            out_dev = find_device(sd, output_name, "output")
        except SetupError:
            out_dev = pick_output(sd, None)
        h.local.new_output()
        self.output = SoundDeviceOutput(sd, out_dev, h.rate, h.local.render)
        if restart_input:
            in_dev = find_device(sd, self.input_query, "input")
            self.source = SoundDeviceInput(sd, in_dev, h.rate, h.channels, h.on_block)
            self.source.start()

    def set_delay(self, delay_ms: float | None):
        """None = automatic."""
        if delay_ms is None:
            self.auto = self.auto or AutoDelay(self.delay)
        else:
            self.auto = None
            self.fixed_delay = min(AutoDelay.MAX, max(0.04, delay_ms / 1000))
            if self.host:
                self.host.set_delay(self.fixed_delay)

    def stop(self):
        self._stop.set()
        if self.advertiser:
            self.advertiser.stop()
        if self.system:
            self.system.restore()
        for thing in (self.source, self.output):
            if thing:
                try:
                    thing.close()
                except Exception:
                    pass
        if self.host:
            self.host.stop()

    def silent_for(self) -> float:
        last = self.host.last_sound if self.host else None
        return time.monotonic() - max(last or 0.0, getattr(self, "started", time.monotonic()))

    def info(self) -> dict:
        h = self.host
        return {
            "mode": "hosting", "name": self.name, "address": lan_ip(), "port": self.port,
            "click": self.click, "capture": self.source.name if isinstance(self.source, SoundDeviceInput) else None,
            "output": self.output.name if self.output else None,
            "level": round(h.take_level(), 3), "silent_s": round(self.silent_for(), 1),
            "delay_ms": round(h.delay * 1000), "auto_delay": self.auto is not None,
            "group_volume": h.group_volume, "group_volume_by": h.group_volume_by,
            "local": player_info(h.local, self.output) if h.local else None,
            "capture_glitches": getattr(self.source, "overflows", 0),
            "listeners": h.listeners(),
        }

    def status_line(self) -> str:
        h = self.host
        bars = int(min(1.0, h.take_level()) * 8 + 0.999)
        parts = [f"in {'▮' * bars}{'▯' * (8 - bars)}"]
        if h.local:
            parts.append(f"this Mac {sync_word(h.local)}")
        for l in h.listeners():
            if l.get("synced"):
                parts.append(f"{l['name']} in sync ({l.get('wifi_ms') or 0:.0f} ms wifi, "
                             f"{l.get('dropouts') or 0} dropouts)")
            else:
                parts.append(f"{l['name']} syncing")
        if not h.listeners():
            parts.append("waiting for friends")
        parts.append(f"delay {h.delay * 1000:.0f} ms" + (" (auto)" if self.auto else ""))
        return " · ".join(parts)


class JoinSession:
    """Listening: play a host's stream on this Mac."""

    def __init__(self, sd, address: str | None, name: str, output_query: str | None = None,
                 settings: Settings | None = None, port: int = PORT, switch_tool: str | None = None,
                 log=print):
        self.sd, self.address, self.name, self.output_query = sd, address, name, output_query
        self.settings = settings or Settings()
        self.port, self.switch_tool, self.log = port, switch_tool, log
        self.client = None
        self._stop = threading.Event()

    def start(self):
        sd = self.sd
        self.output_name = sd.query_devices(pick_output(sd, self.output_query))["name"]
        self.system = SystemOutput(self.log, self.switch_tool)
        self.client = Client(self.address, self._open_output, self.port, self.name, self.settings, log=self.log)
        self.client.start()
        threading.Thread(target=self._tick_loop, daemon=True).start()
        return self

    def _open_output(self, rate: int, render):
        sd = self.sd
        try:
            device = find_device(sd, self.output_name, "output")
        except SetupError:
            device = pick_output(sd, self.output_query)
        self.output_name = sd.query_devices(device)["name"]
        return SoundDeviceOutput(sd, device, rate, render)

    def _tick_loop(self):
        while not self._stop.wait(2.0):
            try:
                self._follow_output()
            except Exception as e:
                self.log(f"! couldn't move to the new speaker: {e}")

    def _follow_output(self):
        """Put AirPods in (or pick another speaker) and the music moves with you."""
        if self.output_query or not self.system.tool:
            return
        current = self.system.current()
        changed, self._seen_output = current != getattr(self, "_seen_output", self.output_name), current
        if not current or not changed or current == self.output_name or is_combined(current):
            return
        self.output_name = current
        self.client.reopen_output(before_open=lambda: refresh_portaudio(self.sd))
        self.log(f'Now playing on "{self.output_name}".')

    def stop(self):
        self._stop.set()
        if self.client:
            self.client.stop()

    def info(self) -> dict:
        c = self.client
        return {
            "mode": "listening", "address": self.address, "host_name": c.host_name, "host_version": c.host_version,
            "connection": c.phase, "output": self.output_name, "error": c.error, "problem": c.problem,
            "delay_ms": round(c.delay * 1000) if c.delay is not None else None,
            "group_volume": c.group_volume, "group_volume_by": c.group_volume_by,
            "clock": c.clock.estimate is not None, **c.stats(),
            "incoming_silent_s": (lambda q: None if q is None else round(min(q, 9999), 1))(
                c.player.silent_for() if c.player else None),
        }


# ----------------------------------------------------------------- terminal
def sync_word(p: Player) -> str:
    if p.read_n is None:
        return "waiting"
    if p.starving:
        return f"no audio arriving ({p.dropouts} dropouts)"
    word = f"in sync (±{max(abs(p.err) * 1000, 0.1):.1f} ms)" if p.in_sync else f"syncing ({p.err * 1000:+.1f} ms)"
    return word + (f", {p.dropouts} dropouts" if p.dropouts else "")


def settings_word(s: Settings) -> str:
    ch = {"both": "L+R", "left": "left only", "right": "right only"}[s.channel]
    return f"vol {s.volume * 100:.0f}% · nudge {s.offset_ms:+.0f} ms · {ch}"


class Console:
    """One live status line plus single-key controls."""

    HELP = "keys: +/- volume · [ ] nudge 1 ms ({ } 10 ms) · c left/right/both · q quit"

    def __init__(self, settings: Settings | None):
        self.settings = settings
        self.tty = sys.stdin.isatty() and sys.stdout.isatty()
        self.quit = threading.Event()
        self.lock = threading.Lock()
        self.line = ""
        self._last_plain = 0.0

    def __enter__(self):
        if self.tty:
            import termios
            import tty
            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            threading.Thread(target=self._keys, daemon=True).start()
        return self

    def __exit__(self, *exc):
        if self.tty:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.flush()

    def _keys(self):
        while not self.quit.is_set():
            if select.select([self.fd], [], [], 0.2)[0]:
                self.key(os.read(self.fd, 1).decode(errors="ignore"))

    def key(self, ch: str):
        s = self.settings
        if ch in "qQ":
            self.quit.set()
        elif s is None:
            return
        elif ch in "+=":
            s.volume = min(1.0, round(s.volume + 0.1, 2))
        elif ch in "-_":
            s.volume = max(0.0, round(s.volume - 0.1, 2))
        elif ch in "[]{}":
            s.offset_ms += {"[": -1, "]": 1, "{": -10, "}": 10}[ch]
        elif ch in "cC":
            s.channel = {"both": "left", "left": "right", "right": "both"}[s.channel]

    def log(self, msg: str):
        with self.lock:
            if self.tty:
                sys.stdout.write(f"\r\x1b[2K{msg}\n{self.line}")
            else:
                sys.stdout.write(msg + "\n")
            sys.stdout.flush()

    def status(self, text: str):
        if self.settings is not None:
            text = f"{text} · {settings_word(self.settings)}"
        with self.lock:
            if self.tty:
                cols = shutil.get_terminal_size((120, 20)).columns
                self.line = "▶ " + text[:cols - 3]
                sys.stdout.write("\r\x1b[2K" + self.line)
                sys.stdout.flush()
            elif time.monotonic() - self._last_plain > 10:
                self._last_plain = time.monotonic()
                print(text, flush=True)


def computer_name() -> str:
    try:
        name = subprocess.run(["scutil", "--get", "ComputerName"], capture_output=True,
                              text=True, timeout=2).stdout.strip()
        if name:
            return name
    except (OSError, subprocess.SubprocessError):
        pass
    return socket.gethostname().removesuffix(".local")


def lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def run_host(args):
    sd = import_sounddevice()
    settings = Settings(args.offset, channel=args.channel)
    with Console(None if args.no_local else settings) as con:
        session = HostSession(sd, args.name or computer_name(), args.delay, args.click, not args.no_local,
                              args.input, args.output, settings, args.port, not args.no_switch, log=con.log)
        session.start()
        try:
            ip = lan_ip()
            delay = "automatic delay" if args.delay is None else f"{args.delay:.0f} ms delay"
            con.log(f'Hosting as "{session.name}" at {ip}  ({session.host.rate} Hz, {delay})')
            con.log("  capturing: " + ("metronome test signal" if args.click else session.source.name))
            con.log("  playing on: " + (session.output.name if session.output else "nothing (--no-local)"))
            con.log(f"  friends run: ./syncaudio join      (or ./syncaudio join {ip})")
            if con.tty:
                con.log("  " + con.HELP)
            while not con.quit.wait(0.25):
                text = session.status_line()
                if not args.click and session.silent_for() > 15:
                    text += " · silent: play something (or allow Terminal in Privacy → Microphone)"
                con.status(text)
        finally:
            session.stop()


def run_client(args):
    sd = import_sounddevice()
    settings = Settings(args.offset, channel=args.channel)
    with Console(settings) as con:
        session = JoinSession(sd, args.host, args.name or computer_name(), args.output, settings,
                              args.port, log=con.log).start()
        con.log(f"Playing on: {session.output_name}")
        con.log(f"Looking for the host at {args.host}..." if args.host else "Looking for the host on this network...")
        if con.tty:
            con.log("  " + con.HELP)
        try:
            while not con.quit.wait(0.25):
                con.status(session.client.status())
        finally:
            session.stop()


def run_devices(_args):
    print(import_sounddevice().query_devices())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="syncaudio", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--output", help="speaker to play on (name or number from `syncaudio devices`)")
        p.add_argument("--offset", type=float, default=0.0, metavar="MS",
                       help="play this Mac MS milliseconds later (negative = earlier)")
        p.add_argument("--channel", choices=["both", "left", "right"], default="both",
                       help="play only one stereo channel (e.g. host left, friend right)")
        p.add_argument("--name", help="name shown to the other Macs")
        p.add_argument("--port", type=int, default=PORT)

    h = sub.add_parser("host", help="share this Mac's audio")
    common(h)
    h.add_argument("--delay", type=float, default=None, metavar="MS",
                   help="fixed capture-to-speaker delay; default: automatic (as low as the Wi-Fi allows)")
    h.add_argument("--input", default="BlackHole", help="device to capture (default: BlackHole)")
    h.add_argument("--click", action="store_true", help="stream a metronome to check the sync by ear")
    h.add_argument("--no-local", action="store_true", help="don't play on this Mac")
    h.add_argument("--no-switch", action="store_true", help="don't change this Mac's sound output")
    h.set_defaults(func=run_host)

    j = sub.add_parser("join", help="play the host's audio on this Mac")
    j.add_argument("host", nargs="?", help="host IP or name.local (default: find it automatically)")
    common(j)
    j.set_defaults(func=run_client)

    sub.add_parser("devices", help="list audio devices").set_defaults(func=run_devices)

    args = ap.parse_args(argv)
    sys.setswitchinterval(0.001)    # hand Python to the audio threads sooner: fewer clicks
    # Quit cleanly (restoring the sound output) when the terminal is closed or killed.
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda *_: sys.exit(0))
    try:
        args.func(args)
    except KeyboardInterrupt:
        pass
    except SetupError as e:
        sys.exit(str(e) + (" Run ./setup.sh host to install it." if e.code == "no_blackhole" else ""))


if __name__ == "__main__":
    main()
