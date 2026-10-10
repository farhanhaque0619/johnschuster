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
import json
import os
import queue
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

PORT = 47800                # TCP carries audio, UDP (same number) clock sync + discovery
MAGIC = "syncaudio/1"
DEFAULT_DELAY_MS = 120      # capture -> speaker; must cover Wi-Fi hiccups + output latency
BLOCK_MS = 5                # capture block size
OUTPUT_LATENCY = 0.04       # requested CoreAudio output buffer; compensated, so it only costs headroom

# --------------------------------------------------------------------- wire
HELLO, HEADER, AUDIO, STATS = b"H", b"C", b"A", b"S"
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
        self.estimate = (t_ref, float(np.median(off - drift * dt)), drift)
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
    adding or skipping single frames now and then; large ones (start-up, Wi-Fi
    stalls) by jumping straight to the right sample.
    """

    HARD_RESYNC = 0.015     # seconds off -> jump
    GAIN = 0.5              # 1/s: how hard small errors are corrected
    MAX_SKEW = 0.002        # never run more than 0.2 % fast or slow
    SMOOTH = 0.05           # error filter, per callback

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
            self.dropouts = 0
            self.resyncs = 0

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

    def _write(self, n: int, pcm: np.ndarray) -> None:
        i, k = n % self.cap, len(pcm)
        first = min(k, self.cap - i)
        self.ring[i:i + first] = pcm[:first]
        self.ring[:k - first] = pcm[first:]

    def _read(self, start: int, need: int) -> np.ndarray:
        out = np.zeros((need, self.channels), np.float32)
        lo = max(start, self.n_first, self.n_end - self.cap)
        hi = min(start + need, self.n_end)
        if hi > lo:
            i, m = lo % self.cap, hi - lo
            first = min(m, self.cap - i)
            out[lo - start:lo - start + first] = self.ring[i:i + first]
            out[lo - start + first:hi - start] = self.ring[:m - first]
        missing = start + need > self.n_end
        if missing and not self.starving:
            self.dropouts += 1          # audio wasn't here in time
        self.starving = missing
        return out

    def render(self, frames: int, dac_local: float, out_channels: int = 2) -> np.ndarray:
        """Audio for `frames` frames whose first frame hits the speaker at dac_local."""
        s = self.settings
        server_t = self.to_server(dac_local)
        with self.lock:
            if server_t is None or self.anchor is None:
                self.read_n = None
                return np.zeros((frames, out_channels), np.float32)
            an, at = self.anchor
            target = an + (server_t - at - s.offset_ms / 1000) * self.rate
            if self.read_n is None or abs(self.read_n - target) > self.HARD_RESYNC * self.rate:
                if self.read_n is not None:
                    self.resyncs += 1
                self.read_n = int(round(target))
                self.acc = self.err = 0.0
            self.err += self.SMOOTH * ((self.read_n - target) / self.rate - self.err)
            skew = min(self.MAX_SKEW, max(-self.MAX_SKEW, -self.GAIN * self.err))
            self.acc += skew * frames
            k = int(self.acc)
            self.acc -= k
            seg = self._read(self.read_n, frames + k)
            self.read_n += frames + k
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
        return self.read_n is not None and not self.starving and abs(self.err) < 0.001


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
def import_sounddevice():
    try:
        import sounddevice
    except (ImportError, OSError) as e:
        sys.exit(f"Can't load the audio library ({e}). Run ./setup.sh first.")
    return sounddevice


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
    raise SystemExit(f'No {kind} device matching "{query}". Available: {names}')


def pick_output(sd, query: str | None) -> int:
    """The named device, else the current default output (unless that is
    BlackHole), else the built-in speakers."""
    if query:
        return find_device(sd, query, "output")
    devices = sd.query_devices()
    default = sd.default.device[1]
    if default is not None and 0 <= default < len(devices) and "blackhole" not in devices[default]["name"].lower():
        return default
    outputs = [i for i, d in enumerate(devices)
               if d["max_output_channels"] > 0 and "blackhole" not in d["name"].lower()]
    speakers = [i for i in outputs if "speaker" in devices[i]["name"].lower()]
    if speakers or outputs:
        return (speakers or outputs)[0]
    raise SystemExit("No speakers found.")


class SoundDeviceOutput:
    def __init__(self, sd, device: int, rate: int, render, now=time.monotonic):
        info = sd.query_devices(device, "output")
        self.name = info["name"]
        self.channels = min(2, info["max_output_channels"])
        failed = []

        def callback(outdata, frames, t, status):
            try:
                lead = t.outputBufferDacTime - t.currentTime
                if not 0.0 <= lead < 1.0:          # timestamps unavailable
                    lead = self.stream.latency
                outdata[:] = render(frames, now() + lead, self.channels)
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
        failed = []

        def callback(indata, frames, t, status):
            try:
                back = t.currentTime - t.inputBufferAdcTime
                if not 0.0 <= back < 1.0:
                    back = frames / rate
                on_block(indata.copy(), now() - back)
            except Exception as e:  # keep capturing whatever happens
                if not failed:
                    failed.append(e)
                    print(f"\ncapture error: {e!r}", file=sys.stderr)

        self.stream = sd.InputStream(samplerate=rate, device=device, channels=channels,
                                     dtype="float32", blocksize=rate * BLOCK_MS // 1000,
                                     latency="low", callback=callback)

    def start(self):
        self.stream.start()

    def close(self):
        self.stream.stop()
        self.stream.close()


# --------------------------------------------------------------------- host
class _Peer:
    QUEUE = 600     # blocks (3 s); a Mac that falls further behind is dropped

    def __init__(self, sock: socket.socket, name: str, ip: str):
        self.sock, self.name, self.ip = sock, name, ip
        self.q: queue.Queue = queue.Queue(maxsize=self.QUEUE)
        self.stats: dict = {}
        self.alive = True

    def push(self, data: bytes):
        try:
            self.q.put_nowait(data)
        except queue.Full:
            self.alive = False


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
        self.running = False

    def start(self):
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp.bind(("", self.port))
        self.tcp.listen()
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp.bind(("", self.port))
        self.running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._udp_loop, daemon=True).start()

    def stop(self):
        self.running = False
        for s in (self.tcp, self.udp):
            try:
                s.close()
            except OSError:
                pass
        with self.lock:
            for p in self.peers:
                p.alive = False

    def on_block(self, pcm: np.ndarray, captured_at: float):
        """Called by the capture device for every block (audio thread)."""
        n, t = self.timeline.stamp(len(pcm), captured_at)
        play_at = t + self.delay
        self.peak = max(self.peak, float(np.abs(pcm).max(initial=0.0)))
        if self.local:
            self.local.feed(n, play_at, pcm)
        with self.lock:
            peers = list(self.peers)
        if peers:
            pcm16 = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
            data = frame(AUDIO, AUDIO_HDR.pack(n, play_at) + pcm16)
            for p in peers:
                p.push(data)

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
            except OSError:
                return
            threading.Thread(target=self._serve, args=(sock, addr[0]), daemon=True).start()

    def _serve(self, sock: socket.socket, ip: str):
        peer = None
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(5)
            kind, payload = recv_frame(sock)
            hello = json.loads(payload)
            if kind != HELLO or hello.get("magic") != MAGIC:
                return
            sock.sendall(frame(HEADER, json.dumps({
                "rate": self.rate, "channels": self.channels,
                "delay": self.delay, "name": self.name}).encode()))
            peer = _Peer(sock, str(hello.get("name") or ip)[:40], ip)
            with self.lock:
                self.peers.add(peer)
            self.log(f"+ {peer.name} joined ({ip})")
            threading.Thread(target=self._read_stats, args=(peer,), daemon=True).start()
            while self.running and peer.alive:
                try:
                    chunks = [peer.q.get(timeout=0.5)]
                except queue.Empty:
                    continue
                while True:
                    try:
                        chunks.append(peer.q.get_nowait())
                    except queue.Empty:
                        break
                sock.sendall(b"".join(chunks))
        except (OSError, ValueError, ConnectionError):
            pass
        finally:
            if peer:
                peer.alive = False
                with self.lock:
                    self.peers.discard(peer)
                if self.running:
                    self.log(f"- {peer.name} left")
            sock.close()

    def _read_stats(self, peer: _Peer):
        try:
            while peer.alive:
                kind, payload = recv_frame(peer.sock)
                if kind == STATS:
                    peer.stats = json.loads(payload)
        except (OSError, ValueError, ConnectionError):
            peer.alive = False

    def status(self) -> str:
        level, self.peak = self.peak, 0.0
        bars = int(min(1.0, level) * 8 + 0.999)
        parts = [f"in {'▮' * bars}{'▯' * (8 - bars)}"]
        if self.local:
            parts.append(f"this Mac {sync_word(self.local)}")
        with self.lock:
            peers = sorted(self.peers, key=lambda p: p.name)
        for p in peers:
            st = p.stats
            if st.get("synced"):
                parts.append(f"{p.name} in sync ({st.get('wifi_ms') or 0:.0f} ms wifi, {st.get('dropouts') or 0} dropouts)")
            else:
                parts.append(f"{p.name} syncing")
        if not peers:
            parts.append("waiting for friends")
        return " · ".join(parts)


# ------------------------------------------------------------------- client
def discover(port: int = PORT, timeout: float = 1.5, address: str = "255.255.255.255"):
    """Find a host on the local network. Returns (ip, name) or None."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            s.sendto(DISCOVER, (address, port))
        except OSError:
            time.sleep(timeout)
            return None
        end = time.monotonic() + timeout
        while (left := end - time.monotonic()) > 0:
            s.settimeout(left)
            try:
                data, addr = s.recvfrom(512)
            except OSError:
                return None
            if data[:1] == b"d":
                try:
                    return addr[0], json.loads(data[1:]).get("name", addr[0])
                except ValueError:
                    pass
    return None


class Client:
    def __init__(self, host: str | None, open_output, port: int = PORT, name: str = "friend",
                 settings: Settings | None = None, log=print, now=time.monotonic,
                 discover_address: str = "255.255.255.255"):
        self.host, self.open_output, self.port, self.name = host, open_output, port, name
        self.settings = settings or Settings()
        self.log, self.now, self.discover_address = log, now, discover_address
        self.clock = ClockSync()
        self.player: Player | None = None
        self.output = None
        self.host_name = None
        self.state = "looking for the host"
        self.leads: deque = deque(maxlen=400)
        self.running = False

    def to_server(self, t: float):
        return self.clock.to_server(t)

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False
        if self.output:
            self.output.close()

    def _run(self):
        while self.running:
            ip = self.host
            if not ip:
                self.state = "looking for the host"
                found = discover(self.port, address=self.discover_address)
                if not found:
                    continue
                ip, self.host_name = found
            try:
                self._session(ip)
            except (OSError, ValueError, KeyError, ConnectionError) as e:
                if self.running and self.state == "playing":
                    self.log(f"lost the host ({e}); reconnecting...")
            except Exception as e:  # e.g. the speaker can't be opened
                self.log(f"error: {e!r}")
                time.sleep(2)
            if self.running:
                self.state = f"can't reach {ip}:{self.port}, retrying" if self.host else "looking for the host"
                time.sleep(1)

    def _session(self, ip: str):
        sock = socket.create_connection((ip, self.port), timeout=3)
        stop = threading.Event()
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.sendall(frame(HELLO, json.dumps({"magic": MAGIC, "name": self.name}).encode()))
            kind, payload = recv_frame(sock)
            if kind != HEADER:
                raise ConnectionError("not a syncaudio host")
            hdr = json.loads(payload)
            rate, channels = int(hdr["rate"]), int(hdr["channels"])
            self.host_name = hdr.get("name", ip)
            self.clock = ClockSync()
            self.leads.clear()
            if self.player is None or (self.player.rate, self.player.channels) != (rate, channels):
                if self.output:
                    self.output.close()
                self.player = Player(rate, channels, self.to_server, self.settings)
                self.output = self.open_output(rate, self.player.render)
            else:
                self.player.reset()
            threading.Thread(target=self._ping_loop, args=(ip, stop), daemon=True).start()
            self.state = "playing"
            self.log(f"connected to {self.host_name} ({ip})")
            last_stats = self.now()
            while self.running:
                kind, payload = recv_frame(sock)
                if kind == AUDIO:
                    n, play_at = AUDIO_HDR.unpack_from(payload)
                    pcm = np.frombuffer(payload, "<i2", offset=AUDIO_HDR.size)
                    self.player.feed(n, play_at, pcm.reshape(-1, channels).astype(np.float32) / 32768)
                    server_now = self.clock.to_server(self.now())
                    if server_now is not None:
                        self.leads.append(play_at - server_now)
                if self.now() - last_stats >= 1.0:
                    last_stats = self.now()
                    sock.sendall(frame(STATS, json.dumps(self.stats()).encode()))
        finally:
            stop.set()
            sock.close()

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
        """How early the tightest recent block arrived, minus output latency."""
        if not self.leads or not self.output:
            return None
        return min(self.leads) - getattr(self.output, "latency", 0.0)

    def stats(self) -> dict:
        p = self.player
        return {
            "synced": bool(p and p.in_sync),
            "err_ms": round(p.err * 1000, 2) if p else None,
            "wifi_ms": round(self.clock.rtt * 1000, 1) if self.clock.rtt is not None else None,
            "dropouts": p.dropouts if p else 0,
        }

    def status(self) -> str:
        if self.state != "playing" or not self.player:
            return self.state
        p, parts = self.player, [f"{self.host_name}"]
        if self.clock.estimate is None:
            parts.append("syncing clocks")
        else:
            parts.append(sync_word(p))
            parts.append(f"wifi {self.clock.rtt * 1000:.1f} ms")
        room = self.headroom()
        if room is not None:
            parts.append(f"headroom {room * 1000:.0f} ms" + (" ⚠ raise --delay on the host" if room < 0.01 else ""))
        return " · ".join(parts)


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


class SystemOutput:
    """Points the Mac's sound output at BlackHole while hosting, and back after."""

    def __init__(self, log):
        self.tool = shutil.which("SwitchAudioSource")
        self.log = log
        self.previous = None

    def _run(self, *args) -> str:
        try:
            return subprocess.run([self.tool, *args, "-t", "output"], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError) as e:
            self.log(f"! couldn't change the sound output: {e}")
            return ""

    def switch_to(self, name: str, fallback: str | None):
        if not self.tool:
            self.log(f'! Set your Mac\'s sound output to "{name}" (Control Center → Sound) '
                     "so apps play into syncaudio. (brew install switchaudio-osx does this for you.)")
            return
        current = self._run("-c")
        self.previous = fallback if "blackhole" in current.lower() else current
        if current != name:
            self._run("-s", name)
            self.log(f'Sound output switched to "{name}" (restored when you quit).')

    def restore(self):
        if self.tool and self.previous:
            self._run("-s", self.previous)
            self.previous = None


def run_host(args):
    sd = import_sounddevice()
    con_settings = Settings(args.offset, channel=args.channel)
    with Console(None if args.no_local else con_settings) as con:
        name = args.name or computer_name()
        out_dev = None if args.no_local else pick_output(sd, args.output)
        if args.click:
            rate, channels, in_dev = 48000, 2, None
        else:
            in_dev = find_device(sd, args.input, "input")
            info = sd.query_devices(in_dev, "input")
            rate, channels = int(info["default_samplerate"]), min(2, info["max_input_channels"])

        host = Host(rate, channels, args.delay / 1000, args.port, name, log=con.log)
        try:
            host.start()
        except OSError as e:
            sys.exit(f"Can't listen on port {args.port} ({e}). Is syncaudio already running?")

        output = system = source = None
        try:
            if out_dev is not None:
                host.local = Player(rate, channels, lambda t: t, con_settings)
                output = SoundDeviceOutput(sd, out_dev, rate, host.local.render)
            if args.click:
                source = ClickSource(rate, channels, host.on_block)
            else:
                source = SoundDeviceInput(sd, in_dev, rate, channels, host.on_block)
                if not args.no_switch:
                    system = SystemOutput(con.log)
                    system.switch_to(source.name, output.name if output else None)
            source.start()

            ip = lan_ip()
            con.log(f'Hosting as "{name}" at {ip}  ({rate} Hz, {args.delay:.0f} ms delay)')
            con.log("  capturing: " + ("metronome test signal" if args.click else source.name))
            con.log("  playing on: " + (output.name if output else "nothing (--no-local)"))
            con.log(f"  friend runs: ./syncaudio join      (or ./syncaudio join {ip})")
            if con.tty:
                con.log("  " + con.HELP)
            started = time.monotonic()
            while not con.quit.wait(0.25):
                text = host.status()
                if not args.click and time.monotonic() - started > 15 and host.timeline.n and \
                        "▮" not in text:
                    text += " · silent: play something (or allow Terminal in Privacy → Microphone)"
                con.status(text)
        except sd.PortAudioError as e:
            sys.exit(f"Audio device error: {e}")
        finally:
            if system:
                system.restore()
            for thing in (source, output):
                if thing:
                    thing.close()
            host.stop()


def run_client(args):
    sd = import_sounddevice()
    settings = Settings(args.offset, channel=args.channel)
    with Console(settings) as con:
        out_dev = pick_output(sd, args.output)
        client = Client(args.host, lambda rate, render: SoundDeviceOutput(sd, out_dev, rate, render),
                        args.port, args.name or computer_name(), settings, log=con.log)
        con.log(f"Playing on: {sd.query_devices(out_dev)['name']}")
        con.log(f"Looking for the host at {args.host}..." if args.host else "Looking for the host on this network...")
        if con.tty:
            con.log("  " + con.HELP)
        client.start()
        try:
            while not con.quit.wait(0.25):
                con.status(client.status())
        finally:
            client.stop()


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
    h.add_argument("--delay", type=float, default=DEFAULT_DELAY_MS, metavar="MS",
                   help=f"capture-to-speaker delay (default {DEFAULT_DELAY_MS}); "
                        "lower = snappier, higher = survives worse Wi-Fi")
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
    # Quit cleanly (restoring the sound output) when the terminal is closed or killed.
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda *_: sys.exit(0))
    try:
        args.func(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
