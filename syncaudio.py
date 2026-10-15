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

The sound itself is handled by audio_engine.py and libsacore (native/sacore.c):
macOS's audio threads only ever run that C code, so nothing Python does - however
busy the app gets - can make them miss a beat.
"""
from __future__ import annotations

import argparse
import errno
import gc
import itertools
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
import traceback
from collections import deque
from typing import NamedTuple

import numpy as np

import audio_engine as ae
import codec
import discovery
import sacore

PORT = 47800                # TCP carries audio, UDP (same number) clock sync + discovery
MAGIC = "syncaudio/1"       # the discovery tag: unchanged, so 1.x and 2.0 Macs still find each other
# The audio protocol. 2.0 can't play with 1.x: a 1.x host can't read 2.0's reports (so it would never
# make room for AirPods), and a 1.x player can't follow 2.0's delay changes, which are per-block stamps.
PROTO = 2
BLOCK_MS = 10               # capture block size
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

    That estimate still wanders (1.x diagnostics showed ramps of 5 ms a second
    and jumps of tens of ms), and the players heard every wander. So what they
    use is a separate, applied mapping: it follows the estimate at most 200 µs
    per second (0.02 % speed, inaudible) and only jumps when the estimate stays
    more than 20 ms away three times in a row (one splice downstream). Its rate
    follows the estimate's freely for the first 30 s, then by at most 10 ppm a
    second: two crystals don't change that fast, but the line fit does whenever
    the offset moves (up to its 500 ppm limit), and that would undo the slew limit.
    """

    MIN_SAMPLES = 8
    WINDOW = 20.0       # seconds of history
    BEST = 0.3          # fraction of pings (lowest round trip) that are used
    MAX_DRIFT = 5e-4    # clocks never disagree by more than 500 ppm
    SLEW = 200e-6       # s/s the applied offset may move
    STEP, STEP_AFTER = 0.020, 3
    DRIFT_SLEW, YOUNG = 10e-6, 30.0     # per s the applied drift may change, once this many s old

    def __init__(self):
        self.samples: deque = deque()
        self.raw = None         # the latest estimate: (t_ref, offset, drift)
        self.estimate = None    # what's applied: (t_a, offset, drift); swapped atomically
        self.rtt = None         # typical round trip of the trusted pings
        self.steps = 0          # times the applied offset jumped
        self.last_step = 0.0
        self.slew = 0.0         # how fast the applied offset moved at the last estimate (s/s)
        self.ping_log: deque = deque(maxlen=200)    # (t4, offset ms, rtt ms, used)
        self._far = 0
        self._born = None

    def add(self, t1: float, t2: float, t3: float, t4: float) -> None:
        rtt = (t4 - t1) - (t3 - t2)
        if rtt < 0:
            return
        mid, off = (t1 + t4) / 2, ((t2 - t1) + (t3 - t4)) / 2
        self.samples.append((mid, off, rtt))
        while mid - self.samples[0][0] > self.WINDOW:
            self.samples.popleft()
        if len(self.samples) < self.MIN_SAMPLES:
            self.ping_log.append((round(t4, 3), round(off * 1000, 3), round(rtt * 1000, 3), False))
            return

        s = np.array(self.samples)
        keep = max(self.MIN_SAMPLES, int(len(s) * self.BEST))
        best = s[np.argsort(s[:, 2])[:keep]]
        t_ref = s[-1, 0]
        dt, offs = best[:, 0] - t_ref, best[:, 1]
        drift = 0.0
        if np.ptp(dt) > 4.0:
            drift = float(np.clip(np.polyfit(dt, offs, 1)[0], -self.MAX_DRIFT, self.MAX_DRIFT))
        offset = float(np.median(offs - drift * dt))
        if self.raw is not None:            # move a fifth of the way: steadier, still follows
            last_ref, last_offset, last_drift = self.raw
            expected = last_offset + last_drift * (t_ref - last_ref)
            if abs(offset - expected) < 0.005:
                offset = expected + 0.2 * (offset - expected)
        self.raw = (t_ref, offset, drift)
        self.rtt = float(np.median(best[:, 2]))
        self.ping_log.append((round(t4, 3), round(off * 1000, 3), round(rtt * 1000, 3), bool(rtt <= best[-1, 2])))
        self._apply(t4)

    def _apply(self, now: float) -> None:
        t_ref, offset, drift = self.raw
        target = offset + drift * (now - t_ref)
        if self.estimate is None:           # the first estimate is taken as it is
            self.estimate, self._born = (now, target, drift), now
            return
        t_a, off_a, drift_a = self.estimate
        cur = off_a + drift_a * (now - t_a)
        self._far = self._far + 1 if abs(target - cur) > self.STEP else 0
        if self._far >= self.STEP_AFTER:    # it really moved (a clock set, a long Wi-Fi detour): jump once
            new, self._far = target, 0
            self.steps += 1
            self.last_step = target - cur
        else:
            most = self.SLEW * max(0.0, now - t_a)
            new = cur + min(most, max(-most, target - cur))
        self.slew = (new - cur) / (now - t_a) if now > t_a else 0.0
        if now - self._born > self.YOUNG:
            most = self.DRIFT_SLEW * max(0.0, now - t_a)
            drift = drift_a + min(most, max(-most, drift - drift_a))
        self.estimate = (now, new, drift)

    def to_server(self, t_local: float):
        est = self.estimate
        if est is None:
            return None
        t_a, offset, drift = est
        return t_local + offset + drift * (t_local - t_a)

    def server_rate(self) -> float:
        """Host seconds per local second."""
        est = self.estimate
        return 1.0 + est[2] if est else 1.0

    def stats(self) -> dict:
        est = self.estimate
        return {"clock_offset_ms": round(est[1] * 1000, 3) if est else None,
                "clock_drift_ppm": round(est[2] * 1e6, 1) if est else None,
                "rtt_ms": round(self.rtt * 1000, 2) if self.rtt is not None else None,
                "clock_slew_ms_s": round(self.slew * 1000, 3), "clock_steps": self.steps}


# ----------------------------------------------------------------- playback
class Settings:
    """Per-Mac knobs, changed live from the keyboard or the app. The sync nudge is kept
    per output device: AirPods and speakers are each off by their own amount."""

    def __init__(self, offset_ms: float = 0.0, volume: float = 1.0, channel: str = "both",
                 offsets: dict | None = None):
        self.offsets = dict(offsets or {})     # device key (UID, or name) -> ms; + plays this Mac later
        self.device_key = None                 # the output playing now
        self.default_offset = offset_ms        # for a device without a nudge of its own
        self.volume = volume
        self.channel = channel                 # both | left | right

    @property
    def offset_ms(self) -> float:
        return self.offsets.get(self.device_key, self.default_offset)

    @offset_ms.setter
    def offset_ms(self, ms: float):
        if self.device_key is None:
            self.default_offset = ms
        else:
            self.offsets[self.device_key] = ms


class Player:
    """What plays on this Mac: the received audio and its stamps (an audio_engine.StreamBuffer),
    the mapping onto the host's clock (identity on the host), and the output engine that plays
    them, attached by audio_engine.AudioOutput. Everything timing-related happens in the engine;
    this keeps the score: the state shown, dropouts, jumps, how early the audio arrives."""

    def __init__(self, rate: int, channels: int, to_server, settings: Settings | None = None,
                 server_rate=lambda: 1.0, now=time.monotonic):
        self.rate, self.channels = rate, channels
        self.to_server, self.server_rate = to_server, server_rate
        self.settings = settings or Settings()
        self.now = now
        self.sb = ae.StreamBuffer(rate, channels)
        self.engine = None
        self.hole_events = 0                    # audio that never arrived, while playing
        self.last_sound = None                  # when a block with actual sound last arrived
        self.reported_s = 0                     # seconds of reports with a headroom in them
        self.report: dict = {}
        self._slack: deque = deque()            # (t, s): how early each block arrived, last 30 s
        self._past = dict.fromkeys(("underruns", "underrun_ms", "skips", "skipped_ms", "splices", "reseeks",
                                    "steps"), 0)    # from outputs this player had before (reopened)
        self._lock = threading.Lock()
        self._closed = False

    def feed(self, n: int, play_at: float, pcm: np.ndarray) -> None:
        with self._lock:
            if self._closed:
                return
            holes = self.sb.holes
            self.sb.feed(n, play_at, pcm)
            e = self.engine
            if self.sb.holes > holes and e is not None and not e.silent:
                self.hole_events += 1
        now = self.now()
        if len(pcm) and float(np.abs(pcm).max()) > 1e-4:
            self.last_sound = now
        t = self.to_server(now)
        if t is not None:
            self._slack.append((now, play_at - t))
            while self._slack[0][0] < now - 30:
                self._slack.popleft()

    def attach(self, engine):
        self.engine = engine

    def detach(self):
        """The output is closing: keep its counts, so they still add up after a reopen."""
        e, self.engine = self.engine, None
        if e is not None:
            for k, v in self._counts(e).items():
                self._past[k] += v

    def _counts(self, e) -> dict:
        c, ms = e.counts, 1000 / self.rate
        return {"underruns": c["underrun_events"], "underrun_ms": c["underrun_frames"] * ms, "skips": c["skips"],
                "skipped_ms": c["skipped_frames"] * ms, "splices": e.splices, "reseeks": e.reseeks, "steps": e.steps}

    def close(self):
        with self._lock:
            self._closed = True
            self.sb.close()

    def _end(self):
        """One past the newest sample arrived (None before any, or once closed: the C ring is gone)."""
        return None if self._closed else self.sb.end()

    def total(self, key: str):
        e = self.engine
        return self._past[key] + (self._counts(e)[key] if e is not None else 0)

    @property
    def dropouts(self) -> int:
        """Audio missing at the speaker: the output ran dry, or the audio never arrived."""
        return self.total("underruns") + self.hole_events

    @property
    def resyncs(self) -> int:
        """Jumps: a crossfade to where the music should be, or a silent restart after running dry."""
        return self.total("splices") + self.total("reseeks")

    @property
    def skips(self) -> int:
        """IO cycles macOS skipped (the static 1.x had)."""
        return self.total("skips")

    @property
    def err(self) -> float:
        """The last timing error measured, seconds on true time (+ = late)."""
        return self.engine.err if self.engine else 0.0

    @property
    def state(self) -> str:
        """waiting (no clock or no audio yet) | no_audio (ran dry in the last second, or the audio is late)
        | syncing | in_sync (held within 3 ms for a second; out beyond 10 ms or at any jump)."""
        e, now = self.engine, self.now()
        if e is None or not e.clock.ready or self.to_server(now) is None or self._end() is None:
            return "waiting"
        if (e.last_underrun is not None and e.now() - e.last_underrun < 1.0) or \
                (e.headroom is not None and e.headroom < 0):
            return "no_audio"
        return "in_sync" if e.sync_since is not None and e.now() - e.sync_since >= 1.0 else "syncing"

    @property
    def in_sync(self) -> bool:
        return self.state == "in_sync"

    def headroom(self):
        """How long the audio that has arrived lasts beyond what plays now (s; negative: late)."""
        return self.engine.headroom if self.engine else None

    def silent_for(self):
        """Seconds since any sound arrived (None before anything arrived)."""
        if self._end() is None:
            return None
        return self.now() - self.last_sound if self.last_sound else float("inf")

    def second(self, output=None) -> dict:
        """This player's numbers for the last second (diagnostics 11.2). The session calls it once a
        second: it also starts the output's next second."""
        rep = output.second() if output else {}
        for k, v in self._past.items():         # counts that carry on across a reopened output
            if k in rep:
                rep[k] = round(rep[k] + v, 1) if isinstance(v, float) else rep[k] + v
        if rep.get("spare_1s_ms") is not None:
            self.reported_s += 1
        now = self.now()
        slack = [s for t, s in tuple(self._slack)]
        last = [s for t, s in tuple(self._slack) if t >= now - 1.0]
        ms = lambda s: round(s * 1000, 1)       # noqa: E731
        rep.update(state=self.state, holes=self.sb.holes, dropouts=self.dropouts, resyncs=self.resyncs,
                   reported_s=self.reported_s, headroom_ms=None if self.headroom() is None else ms(self.headroom()),
                   arrival_slack_min_ms=ms(min(last)) if last else None,
                   arrival_slack_p1_ms=ms(float(np.percentile(slack, 1))) if slack else None)
        self.report = rep
        return rep


# --------------------------------------------------------------- the delay
def ceil_10ms(s: float) -> float:
    return math.ceil(s * 100 - 1e-9) / 100


class Request(NamedTuple):
    mode: str           # urgent | join | raise | lower | manual
    delta: float        # seconds; + = later
    reason: str = ""


class DelaySchedule:
    """When the host's delay changes: on one exact block, so every Mac changes at the same sample
    (each block carries its own play_at). Runs on the capture thread, once per block. A change waits
    for a moment nobody would hear it:
      raise, join, manual   the first silent block after 0.3 s of silence (a gap between songs)
      lower                 only inside a silence longer than what it removes, plus 0.3 s
      urgent                at once: someone is already running out of audio
    A raise that finds no silence in 5 s glides instead: every block 0.25 % later until it's there
    (4 cents; nobody hears it). A join or manual change becomes one short faded step (3-5 s); a lower
    that finds no long enough silence in 10 minutes is dropped, and any other request replaces it."""

    GLIDE = 0.0025
    SILENT_PEAK = 1e-3          # -60 dBFS
    SILENT_RAISE = 0.30
    WAIT = {"urgent": 0.0, "join": 5.0, "raise": 5.0, "manual": 3.0, "lower": 600.0}

    def __init__(self, d: float, on_change=lambda event, **fields: None):
        self.d = d
        self.on_change = on_change          # (event, **fields): delay_change, glide_done, lower_abandoned
        self.pending = None                 # (Request, deadline)
        self.glide_to = None
        self.silent_run = 0.0
        self.changed_at = -math.inf         # when the last step happened (the policy lets reports settle)
        self.lock = threading.Lock()

    @property
    def busy(self) -> bool:
        """A step or a glide on its way. Not a lower: that may wait 10 minutes for its silence."""
        pending = self.pending
        return (pending is not None and pending[0].mode != "lower") or self.glide_to is not None

    @property
    def target(self) -> float:
        """Where the delay is heading (a lower waiting for its silence may never happen: not counted)."""
        pending, glide = self.pending, self.glide_to
        if glide is not None:
            return glide
        return self.d + pending[0].delta if pending and pending[0].mode != "lower" else self.d

    @property
    def lowering(self) -> bool:
        pending = self.pending
        return pending is not None and pending[0].mode == "lower"

    def drop_lower(self, reason: str):
        """The reports no longer allow the lower that is waiting for a silence."""
        with self.lock:
            if self.lowering:
                req, self.pending = self.pending[0], None
                self.on_change("lower_abandoned", by_ms=round(req.delta * 1000, 1), reason=reason)

    def request(self, req: Request, now: float):
        if abs(req.delta) < 1e-6:
            return
        with self.lock:
            if req.mode == "manual":
                self.glide_to = None        # the person's choice wins
            self.pending = (req, now + self.WAIT[req.mode])

    def stamp(self, n: int, captured_at: float, peak: float, block_s: float, now: float) -> float:
        """play_at for the block starting at sample n."""
        with self.lock:
            self.silent_run = self.silent_run + block_s if peak < self.SILENT_PEAK else 0.0
            if self.pending:
                req, deadline = self.pending
                quiet = self.silent_run > (self.SILENT_RAISE if req.delta > 0 else max(0.5, 0.3 - req.delta)) - 1e-6
                if req.mode == "urgent" or quiet or (now >= deadline and req.mode in ("join", "manual")):
                    self._step(n, req, "silence" if quiet and req.mode != "urgent" else req.mode, now)
                elif now >= deadline and req.mode == "raise":
                    self.pending, self.glide_to = None, self.d + req.delta
                    self.on_change("delay_change", from_ms=round(self.d * 1000, 1),
                                   to_ms=round(self.glide_to * 1000, 1), mode="glide", request=req.mode,
                                   reason=req.reason, from_n=n)
                elif now >= deadline:
                    self.pending = None
                    self.on_change("lower_abandoned", by_ms=round(req.delta * 1000, 1), reason=req.reason)
            if self.glide_to is not None:
                most = self.GLIDE * block_s
                self.d += min(most, max(-most, self.glide_to - self.d))
                if abs(self.glide_to - self.d) < 1e-9:
                    self.d, self.glide_to = self.glide_to, None
                    self.on_change("glide_done", to_ms=round(self.d * 1000, 1), from_n=n)
            return captured_at + self.d

    def _step(self, n, req, how, now):
        old, self.d = self.d, self.d + req.delta
        self.pending = self.glide_to = None
        self.changed_at = now
        self.on_change("delay_change", from_ms=round(old * 1000, 1), to_ms=round(self.d * 1000, 1), mode=how,
                       request=req.mode, reason=req.reason, from_n=n)


class DelayPolicy:
    """How long the host's delay should be: long enough that every Mac has each block before it is
    due - through Wi-Fi stalls and busy moments - and no longer. Asked once a second on the host with
    every player's report (its own and each listener's STATS); answers at most one Request for the
    DelaySchedule:
      urgent  a player ran dry or lost audio: up at once (a short faded pause), by what it lacked
      join    a newcomer whose output needs more (AirPods: ~0.2 s to the ear) - before it plays
      raise   over the last 30 s someone came within MARGIN of running dry (a close call is no reason
              for an audible step: this waits for a silence, or glides)
      lower   after 5 calm minutes with plenty spare, and no raise for 10 minutes: a little, in a silence
              (given up as soon as the reports stop allowing it)
    Each player's need is kept as (delay - spare), which doesn't move when the delay does, and reports
    from the first seconds after a change are left out: they still show the old delay. So are reports
    from the seconds after this Mac's own capture stopped (excuse()): everyone plays that gap as
    silence, but the audio after it is as early as ever."""

    MARGIN = 0.060              # spare every player keeps
    STALL_FLOOR = 0.050         # the least a Python stall is allowed
    NET0 = 0.250                # Wi-Fi allowance before anything is measured (1.x saw 110-375 ms stalls)
    D0_MIN, D_MAX = 0.45, 2.0
    LOWER_FLOOR = 0.12          # D_MIN = the longest lead + capture + this
    URGENT_DELTA, URGENT_MIN_GAP = (0.020, 0.300), 5.0
    SETTLE = 5.0
    EXCUSE = 10.0               # s after a capture gap: its silence reaches every ear up to D_MAX later,
                                # and a player's deficit window keeps it 5 s more
    RAISE_AFTER = 10            # reports under the current delay before raising on them
    WINDOW, CALM, NO_LOWER = 30.0, 300.0, 600.0
    LOWER_MAX_STEP = 0.100

    def __init__(self, schedule: DelaySchedule, rate: int = 48000, leads=()):
        self.schedule = schedule
        self.capture_est = ae.IN_BLOCK / rate + 0.010 + 0.005      # capture cycle + block + pump
        # IOProc-to-ear times this Mac's outputs had before (prefs). Not an AirPlay speaker's ~2 s: one
        # session on it is no reason for every share to start that late for a month
        self.leads = [x for x in leads if x and x < 1.0]
        self.players: dict = {}
        self.urgent_at = self.raised_at = self.lowered_at = self.excused_until = -math.inf

    def start_delay(self, lead) -> float:
        """D0: covers the slowest output known (lead), capture, a Wi-Fi stall, a Python stall, margin."""
        need = max([lead or 0.0, *self.leads]) + self.capture_est + self.NET0 + self.STALL_FLOOR + self.MARGIN
        return min(self.D_MAX, ceil_10ms(max(self.D0_MIN, need)))

    def d_min(self, lead=None) -> float:
        leads = [p.lead for p in self.players.values() if p.lead is not None] + self.leads + [lead or 0.0]
        return max(leads) + self.capture_est + self.LOWER_FLOOR

    def clamp(self, d: float, lead=None) -> float:
        return min(self.D_MAX, max(self.d_min(lead), d))

    def excuse(self, now: float):
        """This Mac's capture just stopped for a while (its audio devices were reopened)."""
        self.excused_until = now + self.EXCUSE

    class _Seen:
        def __init__(self):
            self.lead = self.spare = self.under = self.holes = self.rose_at = self.reported = None
            self.deficit = 0.0
            self.joined = None              # the lead its join was judged on
            self.need: deque = deque()      # (t, delay - spare_1s)

    def update(self, now: float, players: dict):
        sch = self.schedule
        d = sch.d
        settled = now - sch.changed_at >= self.SETTLE
        excused = now < self.excused_until
        for key in [k for k in self.players if k not in players]:
            del self.players[key]
        for key, s in players.items():
            p = self.players.setdefault(key, self._Seen())
            if s.get("lead_ms") is not None:
                p.lead = s["lead_ms"] / 1000
            if s.get("reported_s") is None or s["reported_s"] == p.reported:
                continue                    # nothing new from this one
            p.reported = s["reported_s"]
            under, holes = s.get("underruns") or 0, s.get("holes") or 0
            if p.under is not None and (under > p.under or holes > p.holes) and not excused:
                p.rose_at = now
            p.under, p.holes = under, holes
            p.spare = None if s.get("spare_1s_ms") is None else s["spare_1s_ms"] / 1000
            p.deficit = 0.0 if excused else (s.get("deficit_ms") or 0) / 1000
            if p.spare is not None and settled and not excused:
                p.need.append((now, d - p.spare))
            while p.need and p.need[0][0] < now - self.CALM:
                p.need.popleft()
        seen = self.players.values()

        if now - self.urgent_at >= self.URGENT_MIN_GAP and any(
                p.rose_at is not None and p.rose_at > self.urgent_at for p in seen):
            deficit = max(p.deficit for p in seen)            # audio that came too late: that much, and margin
            lacking = max(self.MARGIN - min((p.spare for p in seen if p.spare is not None), default=self.MARGIN),
                          deficit + self.MARGIN if deficit > 0 else 0.0)
            delta = min(self.D_MAX - d, max(self.URGENT_DELTA[0], min(self.URGENT_DELTA[1], lacking)))
            if delta > 0.001:
                self.urgent_at = self.raised_at = now
                return Request("urgent", delta, "ran out of audio")
        if sch.busy:
            return None
        for key, p in self.players.items():
            # a newcomer - or a player whose output now takes longer to the ear (AirPods put in since)
            if p.lead is None or (p.joined is not None and p.lead <= p.joined + 0.020):
                continue
            p.joined = p.lead
            need = p.lead + self.capture_est + self.NET0 + self.STALL_FLOOR + self.MARGIN
            if need > d + 0.010:
                self.raised_at = now
                return Request("join", min(self.D_MAX, ceil_10ms(need)) - d, f"{key}: {p.lead * 1000:.0f} ms to the ear")
        recent = [[r for t, r in p.need if t >= now - self.WINDOW] for p in seen]
        needs = [max(r) for r in recent if len(r) >= self.RAISE_AFTER]
        if needs and max(needs) + self.MARGIN > d + 0.010 and d < self.D_MAX:
            self.raised_at = now
            return Request("raise", min(self.D_MAX, ceil_10ms(max(needs) + self.MARGIN)) - d, "spare under margin")
        def calm(slack):
            """Every player's reports go back the whole calm window, to within `slack` (now and then the
            host's tick sees no new report - it and the listeners' STATS beat - which changes nothing)."""
            return seen and all(p.need and p.need[0][0] <= now - self.CALM + slack for p in seen)
        spare = d - max(max(r for _, r in p.need) for p in seen) if calm(5.0) else None
        if spare is None or spare < 3 * self.MARGIN + 0.10:
            sch.drop_lower("not enough spare any more")         # (someone joined, or the Wi-Fi got worse)
        elif (calm(1.0) and not sch.lowering and now - self.raised_at >= self.NO_LOWER
              and now - self.lowered_at >= self.CALM):
            delta = max(-min(self.LOWER_MAX_STEP, spare - 2 * self.MARGIN), self.d_min() - d)
            if delta <= -0.020:
                self.lowered_at = now
                return Request("lower", delta, f"{spare * 1000:.0f} ms spare for 5 minutes")
        return None


# ------------------------------------------------------------------ capture
def make_click(rate: int, freq: float) -> np.ndarray:
    t = np.arange(int(0.015 * rate)) / rate
    return (np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.003) * 0.7).astype(np.float32)


class ClickSource:
    """A metronome (accent every 4th beat) used instead of BlackHole, so the
    sync can be checked by ear: one crisp click = in sync, a 'flam' = not.
    tone=True (CI): a steady 997 Hz tone on the left at -12 dBFS, clicks on the right,
    so a recorder can check the sound is continuous as well as on time."""

    TONE_HZ, TONE_LEVEL = 997.0, 10 ** (-12 / 20)

    def __init__(self, rate: int, channels: int, on_block, now=time.monotonic, tone: bool = False):
        self.rate, self.channels, self.on_block, self.now, self.tone = rate, channels, on_block, now, tone
        self.hi, self.lo = make_click(rate, 2000), make_click(rate, 1000)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="clicks", daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(1)

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
            click = np.zeros(block, np.float32)
            click[hit] = np.where(beat[hit] % 4 == 0, self.hi[phase[hit]], self.lo[phase[hit]])
            pcm = np.repeat(click[:, None], self.channels, axis=1)
            if self.tone:
                pcm[:, 0] = self.TONE_LEVEL * np.sin(2 * np.pi * self.TONE_HZ * idx / self.rate)
            self.on_block(pcm, n, start + n / self.rate)
            n += block


# ------------------------------------------------------------- audio devices
class SetupError(Exception):
    """A problem the person can fix; `code` tells the app which help to show."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


NO_CORE = ("SyncAudio's audio core is missing. Run ./setup.sh (needs Xcode Command Line Tools: "
           "xcode-select --install).")


def import_sounddevice():
    try:
        import sounddevice
    except (ImportError, OSError) as e:
        sys.exit(f"Can't load the audio library ({e}). Run ./setup.sh first.")
    if not sacore.available():
        sys.exit(NO_CORE)
    return sounddevice


PORTAUDIO = threading.Lock()    # held while PortAudio restarts: listing its devices then would read freed memory


def refresh_portaudio(sd):
    """PortAudio only knows the devices that existed when it started (AirPods
    connected later are invisible). Restart it; no streams may be open."""
    if hasattr(sd, "_terminate"):
        with PORTAUDIO:
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


def device_key(name: str) -> str:
    """How a device is remembered (prefs, sync nudges): its CoreAudio UID, which survives renames
    and reconnects, or its name without the macOS helper."""
    return (sacore.mac_call("device_info", name, True) or {}).get("uid") or name


def output_lead(out) -> float:
    """IOProc-to-ear time of an open output (s): measured once it has called back, else its plan."""
    c = out.engine.clock
    if c.lead is not None:
        return c.lead + c.offset
    return (out.plan.fixed_s or 0.0) + (out.plan.blocksize or ae.OUT_BLOCK) / out.rate


def require_core():
    if not sacore.available():
        raise SetupError("audio", NO_CORE)


# --------------------------------------------------------------------- host
_peer_ids = itertools.count(1)


class _Peer:
    QUEUE = 300     # blocks waiting to be sent (3 s at 10 ms blocks)
    # The kernel's share of the backlog. Audio in there can no longer be dropped when it turns out late
    # (take() only sees the queue); 64 KiB was ~4.4 s of Opus. 16 KiB: about 1 s of Opus, 85 ms of PCM.
    SNDBUF = 16 * 1024
    SLOW_SEND = 0.005

    def __init__(self, sock: socket.socket, name: str, ip: str):
        self.sock, self.name, self.ip = sock, name, ip
        self.id = next(_peer_ids)
        self.q: queue.Queue = queue.Queue(maxsize=self.QUEUE)
        self.stats: dict = {}
        self.version = None
        self.joined = time.monotonic()
        self.alive = True
        self.old = False        # a 1.x listener: told to update, sent no audio
        self.late = 0           # blocks skipped because the network couldn't deliver them in time
        self.codec = "pcm"      # "opus" when both Macs can: a tenth of the Wi-Fi traffic
        self.lead_ms_hint = self.transport = self.engine = None     # from its HELLO
        self.slow_sends: deque = deque(maxlen=50)   # (when, seconds) of sends that waited for Wi-Fi

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

    def send(self, data: bytes):
        """sendall, timed: a send that waits is Wi-Fi (or the listener) not keeping up, not this Mac."""
        t = time.monotonic()
        self.sock.sendall(data)
        took = time.monotonic() - t
        if took > self.SLOW_SEND:
            self.slow_sends.append((t, took))

    def send_stall_max_ms(self, now: float, seconds: float = 5.0):
        return round(max((s for t, s in tuple(self.slow_sends) if t >= now - seconds), default=0.0) * 1000, 1)

    def unsent(self):
        """Bytes the kernel still holds for this listener (not sent, or not yet acknowledged)."""
        try:
            if sys.platform == "darwin":
                return self.sock.getsockopt(socket.SOL_SOCKET, 0x1024)     # SO_NWRITE
            if sys.platform.startswith("linux"):
                import fcntl
                import termios
                return struct.unpack("i", fcntl.ioctl(self.sock.fileno(), termios.TIOCOUTQ, b"\0" * 4))[0]
        except (OSError, ValueError):
            pass
        return None


class Host:
    def __init__(self, rate: int, channels: int, delay: float, port: int = PORT,
                 name: str = "host", log=print, now=time.monotonic, opus: bool = True, stamps: bool = False):
        self.rate, self.channels = rate, channels
        self.port, self.name, self.log, self.now, self.opus = port, name, log, now, opus
        self.events: deque = deque(maxlen=200)
        self.schedule = DelaySchedule(delay, self._delay_changed)
        self.stamps: deque | None = deque(maxlen=100_000) if stamps else None   # (n, play_at) of every block
        self.local: Player | None = None
        self.peers: set = set()
        self.lock = threading.Lock()
        self.peak = 0.0
        self.last_sound = None
        self.group_volume = 1.0     # everyone's volume; any Mac may change it
        self.group_volume_by = None
        self._gain = 1.0            # what was applied to the last block
        self._encoder = None        # Opus, made when the first Mac that can decode it joins
        self._told_old: set = set()
        self.running = False

    @property
    def delay(self) -> float:
        return self.schedule.d

    def event(self, kind: str, **fields):
        self.events.append({"t": round(time.time(), 2), "event": kind, **fields})

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
            peers = [p for p in self.peers if not p.old]
        for p in peers:
            p.push(data)

    def _delay_changed(self, kind: str, **fields):
        """From the DelaySchedule (capture thread): log it, and tell the listeners (for their display;
        what they play follows from the stamps alone)."""
        self.event(kind, **fields)
        if kind != "lower_abandoned":
            self._announce(delay=self.schedule.d, **{k: fields[k] for k in ("from_n", "mode", "to_ms") if k in fields})

    def set_group_volume(self, volume: float, by: str | None = None):
        """Everyone's volume (0..1). Applied here, before the audio is sent."""
        self.group_volume = min(1.0, max(0.0, float(volume)))
        self.group_volume_by = by or self.name
        self._announce(group_volume=self.group_volume, by=self.group_volume_by)

    def on_block(self, pcm: np.ndarray, n: int, captured_at: float):
        """Every captured block (capture thread): n = its first sample's index, captured_at = when that
        sample was captured. Stamped with when it must play, then played here and sent."""
        peak = float(np.abs(pcm).max(initial=0.0))
        self.peak = max(self.peak, peak)
        if peak > 1e-4:
            self.last_sound = time.monotonic()
        play_at = self.schedule.stamp(n, captured_at, peak, len(pcm) / self.rate, self.now())
        if self.stamps is not None:
            self.stamps.append((n, play_at))
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
                # rounded, and x 32768: exactly what the listener's / 32768 undoes
                pcm16 = np.clip(np.rint(pcm * 32768), -32768, 32767).astype("<i2").tobytes()
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
        now = time.monotonic()
        out = []
        for p in peers:
            info = {"state": "waiting", **p.stats, "id": p.id, "name": p.name, "ip": p.ip, "version": p.version,
                    "connected_s": now - p.joined, "late": p.late, "codec": p.codec, "queue": p.q.qsize(),
                    "unsent": p.unsent(), "send_stall_max_ms": p.send_stall_max_ms(now), "engine": p.engine,
                    "transport": p.transport, "old": p.old}
            if info.get("lead_ms") is None:
                info["lead_ms"] = p.lead_ms_hint
            out.append(info)
        return out

    def _udp_loop(self):
        while self.running:
            try:
                data, addr = self.udp.recvfrom(512)
                t2 = self.now()
                if len(data) == PING.size and data[:1] == b"P":
                    _, seq, t1 = PING.unpack(data)
                    self.udp.sendto(PONG.pack(b"Q", seq, t1, t2, self.now()), addr)
                elif data == DISCOVER:
                    self.udp.sendto(b"d" + json.dumps({"name": self.name, "port": self.port, "version": APP_VERSION,
                                                       "proto": PROTO}).encode(), addr)
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
            peer = _Peer(sock, str(hello.get("name") or ip)[:40], ip)
            peer.probe = bool(hello.get("probe"))
            peer.version = hello.get("version")
            peer.old = int(hello.get("proto") or 1) < PROTO
            opus = (self.opus and not peer.old and "opus" in (hello.get("codecs") or [])
                    and codec.available(self.rate, self.channels))
            header = {"rate": self.rate, "channels": self.channels, "delay": self.delay,
                      "group_volume": self.group_volume, "name": self.name, "version": APP_VERSION,
                      "codec": "opus" if opus else "pcm", "proto": PROTO}
            if peer.old:
                # A 1.x listener can't play this. Its own window shows "Different versions ... update"
                # while it's connected, so keep it connected - with no audio - instead of hanging up.
                header["error"] = "update"
            sock.sendall(frame(HEADER, json.dumps(header).encode()))
            peer.codec = None if peer.old else "opus" if opus else "pcm"
            peer.lead_ms_hint, peer.transport, peer.engine = (hello.get(k) for k in ("lead_ms_hint", "transport",
                                                                                      "engine"))
            with self.lock:
                self.peers.add(peer)
            if peer.old:
                self.event("old_listener", name=peer.name, version=peer.version)
                if peer.name not in self._told_old:
                    self._told_old.add(peer.name)
                    self.log(f"! {peer.name} has SyncAudio {peer.version or 'older'}: it needs 2.0 to listen")
            elif not peer.probe:
                self.log(f"+ {peer.name} joined ({ip})")
            threading.Thread(target=self._read_stats, args=(peer,), daemon=True).start()
            while self.running and peer.alive:
                if peer.old:                # something every second, or its connection times out
                    time.sleep(1.0)
                    peer.send(frame(UPDATE, b'{"error": "update"}'))
                    continue
                try:
                    data = peer.take(self.now())
                except queue.Empty:
                    continue
                if data:
                    peer.send(data)
        except (OSError, ValueError, ConnectionError):
            pass
        finally:
            if peer:
                peer.alive = False
                with self.lock:
                    self.peers.discard(peer)
                if self.running and not peer.probe and not peer.old:
                    self.log(f"- {peer.name} left")
            sock.close()

    def _read_stats(self, peer: _Peer):
        try:
            while peer.alive:
                kind, payload = recv_frame(peer.sock)
                if kind == STATS:
                    peer.stats = json.loads(payload)
                elif kind == CONTROL and not peer.old:
                    request = json.loads(payload)
                    if "group_volume" in request:
                        self.set_group_volume(request["group_volume"], by=peer.name)
                        self.log(f"{peer.name} set everyone's volume to {self.group_volume:.0%}")
        except (OSError, ValueError, ConnectionError):
            peer.alive = False


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
        sock.sendall(frame(HELLO, json.dumps({"magic": MAGIC, "name": "connection test", "probe": True,
                                              "proto": PROTO, "version": APP_VERSION}).encode()))
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
    name, version, proto = header.get("name") or address, header.get("version"), header.get("proto")
    result.update(ok=blocks > 0 and bool(rtts), step="done", host_name=name, host_version=version, proto=proto,
                  audio_blocks_per_s=round(blocks / seconds), audio_kbps=round(size * 8 / seconds / 1000),
                  wifi_ms=round(float(np.median(rtts)) * 1000, 1) if rtts else None, pings_answered=len(rtts))
    if result["ok"] and int(proto or 1) < PROTO:        # all fine, but joining would only say "update"
        result.update(ok=False, step="version", problem={
            "code": "old_host", "message": f"{name} is reachable, but has SyncAudio {version or 'older'}. "
            "Update it to 2.0 (Settings → Check for updates)."})
    return result


def player_state(p: Player | None) -> str:
    return p.state if p else "waiting"


class OldHost(ConnectionError):
    """The host runs SyncAudio 1.x, which 2.0 can't play with."""


class Client:
    """Listening to one host: the connection, clock sync, and the Player and output it feeds.
    open_output(rate, channels, player, reason=..., level=...) opens the output (an AudioOutput);
    hint() says what goes in HELLO about it (lead_ms_hint, transport) before it is opened."""

    OLD_HOST_RETRY = 5.0

    def __init__(self, host: str | None, open_output, port: int = PORT, name: str = "friend",
                 settings: Settings | None = None, log=print, now=time.monotonic,
                 discover_address: str | None = None, hint=dict, probes=()):
        self.host, self.open_output, self.port, self.name = host, open_output, port, name
        self.settings = settings or Settings()
        self.log, self.now, self.discover_address, self.hint = log, now, discover_address, hint
        self.probes = probes            # Python's own stalls, sent in each second's STATS
        self.clock = ClockSync()
        self.player: Player | None = None
        self.output = None
        self.host_name = None
        self.host_version = None
        self.codec = None               # how the audio arrives: "opus" or "pcm"
        self.delay = None
        self.group_volume = None
        self.group_volume_by = None
        self.report: dict = {}          # the last second's numbers (sent as STATS)
        self.events: deque = deque(maxlen=200)
        self._sock = None
        self._conn = None               # the connection, from before the host answers (stop() cuts it)
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

    def event(self, kind: str, **fields):
        self.events.append({"t": round(time.time(), 2), "event": kind, **fields})

    def to_server(self, t: float):
        return self.clock.to_server(t)

    def server_rate(self) -> float:
        return self.clock.server_rate()

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False
        conn = self._conn
        if conn:
            try:
                conn.shutdown(socket.SHUT_RDWR)     # the thread waiting on it stops now
            except OSError:
                pass
        with self._output_lock:
            if self.output:
                self.output.close()
                self.output = None
            if self.player:
                self.player.close()

    def reopen_output(self, before_open=None, level: int | None = None, reason: str = "default_changed"):
        """Move playback to whatever device open_output picks now (or reopen it at another IO level)."""
        with self._output_lock:
            if not self.running:            # (stopped meanwhile: nothing to open, and PortAudio left alone)
                return
            if self.output:
                self.output.close()
                self.output = None
            if before_open:
                before_open()
            if self.player:
                self.output = self.open_output(self.player.rate, self.player.channels, self.player,
                                               reason=reason, level=level)

    def watchdog(self):
        with self._output_lock:
            return self.output.watchdog() if self.output else None

    def _run(self):
        while self.running:
            ip, wait = self.host, 1.0
            if not ip:
                self.phase = "searching"
                found = discover(self.port, address=self.discover_address)
                if not found:
                    continue
                ip, self.host_name = found
            try:
                self.phase = "connecting"
                self._session(ip)
            except OldHost:
                wait = self.OLD_HOST_RETRY      # (each try shows up on that Mac as a join)
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
                time.sleep(wait)

    def _session(self, ip: str):
        sock = socket.create_connection((ip, self.port), timeout=3)
        self._conn = sock
        stop = threading.Event()
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            hello = {"magic": MAGIC, "name": self.name, "version": APP_VERSION, "proto": PROTO, "engine": ae.ENGINE,
                     "codecs": ["opus"] if codec.available() else []}
            sock.sendall(frame(HELLO, json.dumps({**hello, **self.hint()}).encode()))
            kind, payload = recv_frame(sock)
            if kind != HEADER:
                raise ConnectionError("not a syncaudio host")
            hdr = json.loads(payload)
            self.host_version = hdr.get("version")      # None: a version from before versions were sent
            self.host_name = hdr.get("name", ip)
            if int(hdr.get("proto") or 1) < PROTO:
                v = self.host_version
                self.problem = {"code": "old_host", "message": f"{self.host_name} has SyncAudio {v or 'older'}. "
                                "Update it to 2.0 (Settings → Check for updates)."}
                self.event("old_host", version=v)
                raise OldHost(self.problem["message"])
            rate, channels = int(hdr["rate"]), int(hdr["channels"])
            self.delay = float(hdr.get("delay", 0))
            self.group_volume = float(hdr.get("group_volume", 1.0))
            self.codec = hdr.get("codec", "pcm")
            decoder = codec.Decoder(channels) if self.codec == "opus" else None
            self.clock = ClockSync()
            with self._output_lock:     # a fresh player and output for every connection
                if not self.running:    # (stopped while the host was answering)
                    return
                if self.output:
                    self.output.close()
                    self.output = None
                if self.player:
                    self.player.close()
                self.player = Player(rate, channels, self.to_server, self.settings, self.server_rate)
                self.output = self.open_output(rate, channels, self.player)
            self.error = self.problem = None
            self._sock = sock
            self._send(STATS, self.second())    # this output's real lead, before any audio is due
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
                    if "mode" in update:        # the host stepped its delay, or started a glide towards to_ms
                        self.event("delay_change", from_ms=round(self.delay * 1000, 1), to_ms=update.get("to_ms"),
                                   mode=update["mode"], from_n=update.get("from_n"))
                    self.delay = float(update.get("delay", self.delay))
                    if "group_volume" in update:
                        self.group_volume = float(update["group_volume"])
                        self.group_volume_by = update.get("by")
                if self.now() - last_stats >= 1.0:
                    last_stats = self.now()
                    self._send(STATS, self.second())
        finally:
            stop.set()
            self._sock = self._conn = None
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

    def second(self) -> dict:
        """This Mac's numbers for the last second (diagnostics 11.2), kept in .report and sent as STATS."""
        with self._output_lock:
            rep = self.player.second(self.output) if self.player else {}
        before = self.report
        rep.update(self.clock.stats(), engine=ae.ENGINE)
        for probe in self.probes:
            rep.update(probe.second())
        if rep["clock_steps"] > before.get("clock_steps", 0):
            self.event("clock_step", ms=round(self.clock.last_step * 1000, 1))
        note_second(self.event, before, rep)
        self.report = rep
        return self.stats()

    def stats(self) -> dict:
        p = self.player
        room = self.headroom()
        live = {"state": player_state(p), "synced": bool(p and p.in_sync),
                "wifi_ms": round(self.clock.rtt * 1000, 1) if self.clock.rtt is not None else None,
                "headroom_ms": round(room * 1000) if room is not None else None,
                "dropouts": p.dropouts if p else 0, "resyncs": p.resyncs if p else 0,
                "skips": p.skips if p else 0, "glitches": glitches(p), "codec": self.codec}
        out = {**self.report, **live}
        if out.get("err_ms") is None and p:
            out["err_ms"] = round(p.err * 1000, 3)
        if out.get("lead_ms") is None and self.output:      # the very first STATS: before a second has passed
            out["lead_ms"] = round(output_lead(self.output) * 1000, 1)
        return out

    def status(self) -> str:
        if self.phase != "connected" or not self.player:
            return self.problem["message"] if self.problem else self.state
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


def note_second(event, before: dict, rep: dict):
    """Events worth keeping from one second to the next: macOS skipping audio, the ring running dry,
    the device clock's line starting again, the device saying its latency changed (which explains a
    skip or a new line at that moment)."""
    if rep.get("skips", 0) > before.get("skips", 0):
        event("skip_burst", count=rep["skips"] - before.get("skips", 0),
              ms=round(rep.get("skipped_ms", 0) - before.get("skipped_ms", 0), 1))
    if rep.get("underruns", 0) > before.get("underruns", 0):
        event("underrun", ms=round(rep.get("underrun_ms", 0) - before.get("underrun_ms", 0), 1))
    if rep.get("dll_reanchors", 0) > before.get("dll_reanchors", 0):
        event("dll_reanchor")
    if rep.get("latency_changes", 0) > before.get("latency_changes", 0):
        event("latency_change", count=rep["latency_changes"] - before.get("latency_changes", 0))


# ----------------------------------------------------------------- sessions
class SystemOutput:
    """The Mac's sound output: pointed at BlackHole while hosting, back after; and the output to follow.

    Through libsacore's macOS helper when it's there (no processes, and it is told when the output
    changes); otherwise through SwitchAudioSource, started with posix_spawn (an absolute path and
    close_fds=False make CPython use it) instead of a fork of the whole app."""

    def __init__(self, log, tool: str | None = None):
        self.tool = tool or shutil.which("SwitchAudioSource")
        self.log = log
        self.previous = None
        self.counter = None
        if sacore.mac_call("default_output"):
            sel = sacore.mac.SEL
            self.counter = sacore.mac_call("listen", sacore.mac.system, sel["default_output"], sel["global"])
        self._seen = self.counter.value if self.counter else 0

    @property
    def available(self) -> bool:
        return bool(self.counter or self.tool)

    def _run(self, *args) -> str:
        try:
            return subprocess.run([self.tool, *args, "-t", "output"], capture_output=True, text=True,
                                  timeout=5, close_fds=False).stdout.strip()
        except (OSError, subprocess.SubprocessError) as e:
            self.log(f"! couldn't change the sound output: {e}")
            return ""

    def current(self) -> str:
        if self.counter:
            dev = sacore.mac_call("default_output")
            name = (sacore.mac_call("device_info_id", dev, True) or {}).get("name") if dev else None
            if name:
                return name
        return self._run("-c") if self.tool else ""

    def changed(self) -> bool:
        """Whether macOS said the output changed since the last call (False without the helper)."""
        if not self.counter:
            return False
        value = self.counter.value
        changed, self._seen = value != self._seen, value
        return changed

    def _set(self, name: str):
        info = sacore.mac_call("device_info", name, True) if self.counter else None
        if not (info and sacore.mac_call("set_default_output", info["id"])) and self.tool:
            self._run("-s", name)

    def switch_to(self, name: str, fallback: str | None):
        if not self.available:
            self.log(f'! Set this Mac\'s sound output to "{name}" (Control Center → Sound) '
                     "so apps play into SyncAudio.")
            return
        current = self.current()
        self.previous = fallback if is_combined(current) else current
        if current != name:
            self._set(name)
            self.log(f'Sound output switched to "{name}" (restored when you stop).')

    def set(self, name: str):
        if self.available and name:
            self._set(name)

    def restore(self, fallback: str | None = None):
        """Back to the output from before sharing - or, if that one has gone meanwhile (AirPods back in
        their case), to `fallback`: never left on BlackHole, where no app can be heard."""
        if self.available and self.previous:
            self._set(self.previous)
            if fallback and fallback != self.previous and self.current() != self.previous:
                self._set(fallback)
            self.previous = None

    def close(self):
        if self.counter:
            self.counter.close()
            self.counter = None


def glitches(p: Player | None, capture: dict | None = None) -> int:
    """Every audible hiccup: audio missing or late, jumps, macOS skipping, capture falling behind."""
    return (p.dropouts + p.resyncs + p.skips if p else 0) + (capture or {}).get("cap_overflows", 0)


def player_info(p: Player | None, capture: dict | None = None) -> dict:
    """A player's state and its last second's numbers (diagnostics 11.2)."""
    if p is None:
        return {"state": "waiting"}
    return {**p.report, "state": p.state, "dropouts": p.dropouts, "resyncs": p.resyncs, "skips": p.skips,
            "glitches": glitches(p, capture),
            "err_ms": p.report.get("err_ms", round(p.err * 1000, 3))}


class _Session:
    """What hosting and listening share: remembering devices, keeping track of opened outputs, and
    acting on what an output's watchdog says."""

    TICK = 0.5

    def _device(self, name: str, uid: str | None = None) -> dict:
        """This device's entry in the remembered devices (the app's prefs["devices"])."""
        dev = self.devices.setdefault(uid or device_key(name), {})
        dev.update(name=name, seen=round(time.time()))
        return dev

    def _uid(self, out):
        return (out.facts or {}).get("uid")

    def _opened(self, out, event):
        """Bookkeeping for a newly opened output: its nudge applies from now on."""
        dev = self._device(out.name, self._uid(out))
        dev["uid"] = self._uid(out)
        self.settings.device_key = self._uid(out) or device_key(out.name)
        event("output_open", name=out.name, uid=dev["uid"], reason=out.reason, level=out.plan.level)
        if out.call_profile:
            event("call_profile", name=out.name)

    def _remember(self, out, report):
        if out is not None and report.get("lead_ms") is not None:
            self._device(out.name, self._uid(out))["lead_ms"] = report["lead_ms"]

    def _act(self, act, out, event, reopen):
        """An output watchdog's action. reopen(level, reason, follow)."""
        if act["action"] == "reopen":
            if act["persist"]:
                self._device(out.name, self._uid(out))["io_level"] = act["level"]
            event("plan_level", **{"from": act["from"], "to": act["level"], "reason": act["reason"]})
            self.log(f"audio: reopening {out.name} at IO level {act['level']} ({act['reason']})")
            reopen(act["level"], act["reason"], False)
        elif act["action"] == "follow":
            self.log(f'audio: "{out.name}" stopped ({act["reason"]}); following the Mac\'s output')
            reopen(None, act["reason"], True)
        elif act["action"] == "warn":
            event(act["warning"], count=act.get("count"))


class HostSession(_Session):
    """Everything hosting needs: capture, playback on this Mac, the network
    side, switching the sound output to BlackHole, and the automatic delay."""

    def __init__(self, sd, name: str, delay_ms: float | None = None, click: bool = False,
                 play_local: bool = True, input_query: str = "BlackHole", output_query: str | None = None,
                 settings: Settings | None = None, port: int = PORT, switch_output: bool = True,
                 switch_tool: str | None = None, log=print, devices: dict | None = None,
                 opus: bool = True, tone: bool = False, stamps: bool = False, peer_leads: dict | None = None,
                 probes=()):
        self.sd, self.name, self.click, self.play_local = sd, name, click, play_local
        self.input_query, self.output_query, self.port = input_query, output_query, port
        self.settings = settings or Settings()
        self.switch_output, self.switch_tool, self.log = switch_output, switch_tool, log
        self.devices = {} if devices is None else devices      # remembered per device (the app's prefs)
        self.peer_leads = {} if peer_leads is None else peer_leads     # listeners' leads, by name (prefs too)
        self.probes = probes        # Python's own stalls (audio_engine.GcTimer, GilProbe): in this Mac's report
        self.opus, self.tone, self.keep_stamps = opus, tone, stamps
        self.auto = delay_ms is None
        self.fixed_delay = None if delay_ms is None else delay_ms / 1000
        self.host = self.local = self.output = self.source = self.system = self.advertiser = self.policy = None
        self.local_report: dict = {}
        self.cap_report: dict = {}
        self._seen_output = None
        self._retry_at = None           # this Mac's output couldn't be opened: when to try again
        self._capture_retry = None      # ...or its capture: when to try that again
        self._lock = threading.RLock()
        self._stop = threading.Event()

    @property
    def delay(self) -> float:
        return self.host.delay if self.host else (self.fixed_delay or DelayPolicy.D0_MIN)

    def event(self, kind: str, **fields):
        if self.host:
            self.host.event(kind, **fields)

    def start(self):
        require_core()
        sd = self.sd
        out_dev = pick_output(sd, self.output_query) if self.play_local else None
        if self.click:
            rate, channels, in_dev = 48000, 2, None
        else:
            in_dev = find_device(sd, self.input_query, "input")
            info = sd.query_devices(in_dev, "input")
            rate, channels = int(info["default_samplerate"]), min(2, info["max_input_channels"])
        self.host = Host(rate, channels, DelayPolicy.D0_MIN, self.port, self.name, log=self.log, opus=self.opus,
                         stamps=self.keep_stamps)
        try:
            self.host.start()
        except OSError as e:
            self.host = None
            raise SetupError("port_busy", f"Can't share on port {self.port} ({e}). "
                                          "Is SyncAudio already sharing on this Mac?")
        try:
            # outputs this Mac and its listeners played on in the last 30 days: AirPods used yesterday count
            # from the start, so nobody's first song stops for a step up when they join
            month = time.time() - 30 * 86400
            leads = [d["lead_ms"] / 1000 for d in [*self.devices.values(), *self.peer_leads.values()]
                     if d.get("lead_ms") and d.get("seen", 0) > month]
            self.policy = DelayPolicy(self.host.schedule, rate, leads)
            if out_dev is not None:
                self.local = Player(rate, channels, lambda t: t, self.settings)
                self.output = self._open_output(out_dev, "start")
                self.host.local = self.local
            lead = output_lead(self.output) if self.output else None
            # Before the first block is stamped: the start delay covers this Mac's own output already
            self.host.schedule.d = self.policy.start_delay(lead) if self.auto else self.policy.clamp(self.fixed_delay, lead)
            self.system = SystemOutput(self.log, self.switch_tool)
            if self.click:
                self.source = ClickSource(rate, channels, self.host.on_block, tone=self.tone)
                self.source.start()
            else:
                self.source = ae.AudioInput(sd, in_dev, rate, channels, self.host.on_block, log=self.log)
                if self.switch_output:
                    self.system.switch_to(self.source.name, self.output.name if self.output else None)
        except getattr(sd, "PortAudioError", ()) as e:
            self.stop()
            raise SetupError("audio", f"Couldn't open the audio device ({e}).")
        except BaseException:
            self.stop()
            raise
        self.started = time.monotonic()
        # Never over the AirDrop link while sharing: it makes Wi-Fi hop channels and stall (dropouts).
        self.advertiser = discovery.Advertiser(self.name, self.port, lan_ip(), p2p_seconds=0).start()
        threading.Thread(target=self._tick_loop, name="host-tick", daemon=True).start()
        return self

    def _open_output(self, index: int, reason: str, level: int | None = None):
        name = self.sd.query_devices(index)["name"]
        out = ae.AudioOutput(self.sd, index, self.host.rate, self.local, self.settings, self._device(name),
                             log=self.log, reason=reason, level=level)
        self._opened(out, self.event)
        return out

    def _tick_loop(self):
        ticks, failed = 0, False
        while not self._stop.wait(self.TICK):
            ticks += 1
            try:
                if self.system.changed() or (not self.system.counter and ticks % 4 == 0):
                    self._follow_output()
                if ticks % 2 == 0:
                    self._second()
            except Exception as e:      # keep going: this loop is what follows devices and sets the delay
                if not failed:
                    failed = True
                    self.log(f"! host tick: {e!r}\n{traceback.format_exc()}")

    def _second(self):
        with self._lock:
            if self._stop.is_set():
                return
            if self.play_local and self.output is None and time.monotonic() >= (self._retry_at or math.inf):
                self._open_again()
            if self.output:
                before = self.local_report
                self.local_report = self.local.second(self.output)
                for probe in self.probes:
                    self.local_report.update(probe.second())
                note_second(self.event, before, self.local_report)
                self._remember(self.output, self.local_report)
                act = self.output.watchdog()
                if act:
                    self._act(act, self.output, self.event, self._reopen_output)
            if isinstance(self.source, ae.AudioInput):
                self._watch_capture(time.monotonic())
        listeners = [l for l in self.host.listeners() if not l["old"]]
        for l in listeners:
            if l.get("lead_ms") is not None:
                self.peer_leads[l["name"]] = {"lead_ms": l["lead_ms"], "seen": round(time.time())}
        if self.auto:
            players = {"this Mac": self.local_report} if self.output else {}
            players.update((l["id"], l) for l in listeners)
            req = self.policy.update(time.monotonic(), players)
            if req:
                self.host.schedule.request(req, time.monotonic())
                self.log(f"delay: {req.mode} {req.delta * 1000:+.0f} ms ({req.reason})")

    def _follow_output(self):
        """AirPods connecting (or picking another speaker) moves the Mac's sound
        output away from BlackHole. Play this Mac's share there instead, and
        point the output back at BlackHole so everyone keeps hearing the apps."""
        if not (self.system and self.system.available) or self._stop.is_set():
            return
        current = self.system.current()
        changed, self._seen_output = current != self._seen_output, current
        if not current or current.lower().startswith("blackhole") or not changed:
            return
        playing = self.output.name if self.output else None
        if self.play_local and not self.output_query and current != playing and not is_combined(current):
            self._reopen_audio(current, "default_changed")
            if self.output:
                self.log(f'This Mac now plays on "{current}".')
        with self._lock:            # (not once Stop has given the Mac its output back)
            if self._stop.is_set() or self.click or not self.switch_output or \
                    not isinstance(self.source, ae.AudioInput):
                return
            if not is_combined(current):
                self.system.previous = current
            self.system.set(self.source.name)
        self.log(f'Sound output was "{current}"; put back on "{self.source.name}" so everyone hears it.')

    def _watch_capture(self, now: float):
        """BlackHole: reopened when it stops calling back, and tried again every few seconds while it
        can't be opened. Its own gaps are no reason for a longer delay (DelayPolicy.excuse)."""
        if self._capture_retry is not None:     # down: everyone hears silence until it's back
            self.policy.excuse(now)
            if now >= self._capture_retry:      # (with PortAudio restarted: it may be back as another device)
                self._reopen_audio(self._follow_target() if self.play_local else None, "capture_retry")
            return
        self.cap_report = self.source.second()
        if self.source.missed():
            self.policy.excuse(now)
        if self.source.watchdog():
            self.log("audio: the capture device stopped calling back; reopening it")
            self._reopen_input()

    def _follow_target(self):
        """Where this Mac plays once the output it played on is gone: --output's device, else the Mac's
        sound output - but while sharing that is BlackHole (its own capture: an echo for everyone), so
        then the output it had before, which _follow_output keeps up to date. None: pick_output's."""
        if self.output_query:
            return self.output_query
        current = self.system.current() if self.system and self.system.available else ""
        if current and not is_combined(current):
            return current
        return self.system.previous if self.system else None

    def _output_failed(self, e: Exception):
        """This Mac's output couldn't be opened (a Bluetooth device still connecting, say): everyone
        else plays on, and it is tried again every few seconds (and on the next output change)."""
        if self._retry_at is None:
            self.log(f"! couldn't play on this Mac ({e}); trying again every few seconds")
            self.event("output_failed", error=str(e))
        self._retry_at = time.monotonic() + 3.0

    def _open_again(self):
        """Without restarting PortAudio, which would stop capture for everyone."""
        with self._lock:
            try:
                name = self._follow_target()
                try:
                    index = find_device(self.sd, name, "output") if name else pick_output(self.sd, None)
                except SetupError:
                    index = pick_output(self.sd, None)
                self.output = self._open_output(index, "retry")
                self._retry_at = None
                self._playing_on(self.output.name)
            except Exception as e:      # noqa: BLE001 - keep trying
                self._output_failed(e)

    def _playing_on(self, name: str):
        """This Mac followed its output to `name`: that is what Stop gives the sound back to (the one it
        had before may be gone - AirPods back in their case)."""
        if self.system and not self.output_query:
            self.system.previous = name

    def _reopen_output(self, level, reason, follow):
        if follow:
            self._reopen_audio(self._follow_target(), reason)
            return
        with self._lock:
            index = self.output.index
            self.output.close()
            self.output = None
            try:
                self.output = self._open_output(index, reason, level)
            except Exception as e:      # noqa: BLE001 - see _output_failed
                self._output_failed(e)

    def _reopen_audio(self, output_name: str | None, reason: str):
        """Close everything, restart PortAudio (so it sees new devices), reopen: capture carries on
        with its sample index where real time says, so the gap is one counted silence."""
        with self._lock:
            if self._stop.is_set():
                return
            sd = self.sd
            capture = isinstance(self.source, ae.AudioInput)
            after = self.source.continuity() if capture else None
            for thing in ([self.source] if capture else []) + [self.output]:
                try:
                    if thing:
                        thing.close()
                except Exception:
                    pass
            self.output = None
            refresh_portaudio(sd)
            if self.play_local:
                try:
                    try:
                        out_dev = find_device(sd, output_name, "output") if output_name else pick_output(sd, None)
                    except SetupError:
                        out_dev = pick_output(sd, None)
                    self.output = self._open_output(out_dev, reason)
                    self._retry_at = None
                    self._playing_on(self.output.name)
                except Exception as e:      # noqa: BLE001 - see _output_failed
                    self._output_failed(e)
            if capture:                 # whatever happened to this Mac's output, everyone keeps hearing
                self._open_input(after)

    def _reopen_input(self):
        after = self.source.continuity()
        self.source.close()
        self._open_input(after)

    def _open_input(self, after):
        """Capture from BlackHole again, n carrying on where real time says (`after`). If it can't be
        opened (coreaudiod restarting, say), it is tried again every few seconds (_watch_capture)."""
        try:
            in_dev = find_device(self.sd, self.input_query, "input")
            self.source = ae.AudioInput(self.sd, in_dev, self.host.rate, self.host.channels, self.host.on_block,
                                        after=after, log=self.log)
            self._capture_retry = None
        except Exception as e:      # noqa: BLE001 - keep trying; self.source stays the closed one
            if self._capture_retry is None:
                self.log(f"! couldn't capture from {self.input_query} ({e}); trying again every few seconds")
                self.event("capture_failed", error=str(e))
            self._capture_retry = time.monotonic() + 3.0
        self.policy.excuse(time.monotonic())        # everyone plays the gap as silence: not their fault

    def set_delay(self, delay_ms: float | None):
        """None = automatic. A number: that delay from the next quiet moment (3 s at most), kept."""
        self.auto = delay_ms is None
        if delay_ms is not None:
            self.fixed_delay = delay_ms / 1000
            if self.host:
                target = self.policy.clamp(self.fixed_delay)
                self.host.schedule.request(Request("manual", target - self.host.delay, "set by hand"),
                                           time.monotonic())

    def stop(self):
        self._stop.set()
        with self._lock:
            if self.advertiser:
                self.advertiser.stop()
            if self.system:
                try:        # if the output it had before is gone: where this Mac plays, else its speakers
                    fallback = (self.output.name if self.output else
                                self.sd.query_devices(pick_output(self.sd, None))["name"])
                except Exception:       # noqa: BLE001 - only a fallback
                    fallback = None
                self.system.restore(fallback)
                self.system.close()
            for thing in (self.source, self.output):
                if thing:
                    try:
                        thing.close()
                    except Exception:
                        pass
            if self.local:
                self.local.close()
            if self.host:
                self.host.stop()

    def silent_for(self) -> float:
        last = self.host.last_sound if self.host else None
        return time.monotonic() - max(last or 0.0, getattr(self, "started", time.monotonic()))

    def info(self) -> dict:
        h, out, src = self.host, self.output, self.source
        sch = h.schedule
        capture = isinstance(src, ae.AudioInput)
        return {
            "mode": "hosting", "name": self.name, "address": lan_ip(), "port": self.port,
            "click": self.click, "output": out.name if out else None,
            "capture": {"name": src.name, "input": src.info(), **self.cap_report} if capture else None,
            "level": round(h.take_level(), 3), "silent_s": round(self.silent_for(), 1),
            "delay_ms": round(sch.d * 1000), "delay_target_ms": round(sch.target * 1000),
            "glide": sch.glide_to is not None, "auto_delay": self.auto,
            "group_volume": h.group_volume, "group_volume_by": h.group_volume_by,
            "local": player_info(self.local, self.cap_report) if self.local else None,
            "capture_glitches": self.cap_report.get("cap_overflows", 0) + self.cap_report.get("cap_skips", 0),
            "listeners": h.listeners(),
            "audio": {"output": out.info() if out else None},
            "events": list(h.events),
        }

    def status_line(self) -> str:
        h = self.host
        bars = int(min(1.0, h.take_level()) * 8 + 0.999)
        parts = [f"in {'▮' * bars}{'▯' * (8 - bars)}"]
        if self.local:
            parts.append(f"this Mac {sync_word(self.local)}")
        listeners = [l for l in h.listeners()]
        for l in listeners:
            if l["old"]:
                parts.append(f"{l['name']} needs SyncAudio 2.0")
            elif l.get("state") == "in_sync":
                parts.append(f"{l['name']} in sync ({l.get('wifi_ms') or 0:.0f} ms wifi, "
                             f"{l.get('dropouts') or 0} dropouts)")
            else:
                parts.append(f"{l['name']} syncing")
        if not listeners:
            parts.append("waiting for friends")
        sch = h.schedule
        parts.append(f"delay {sch.d * 1000:.0f} ms" + (" (auto)" if self.auto else "")
                     + (f" → {sch.target * 1000:.0f}" if sch.busy else ""))
        return " · ".join(parts)


class JoinSession(_Session):
    """Listening: play a host's stream on this Mac."""

    def __init__(self, sd, address: str | None, name: str, output_query: str | None = None,
                 settings: Settings | None = None, port: int = PORT, switch_tool: str | None = None,
                 log=print, devices: dict | None = None, probes=()):
        self.sd, self.address, self.name, self.output_query = sd, address, name, output_query
        self.settings = settings or Settings()
        self.port, self.switch_tool, self.log = port, switch_tool, log
        self.devices = {} if devices is None else devices
        self.probes = probes
        self.client = None
        self._failing = False           # this Mac's output couldn't be opened: tried again every few seconds
        self._stop = threading.Event()

    def start(self):
        require_core()
        sd = self.sd
        self.output_name = sd.query_devices(pick_output(sd, self.output_query))["name"]
        self._seen_output = self.output_name
        self.system = SystemOutput(self.log, self.switch_tool)
        self.client = Client(self.address, self._open_output, self.port, self.name, self.settings, log=self.log,
                             hint=self._hint, probes=self.probes)
        self.client.start()
        threading.Thread(target=self._tick_loop, name="join-tick", daemon=True).start()
        return self

    def event(self, kind: str, **fields):
        if self.client:
            self.client.event(kind, **fields)

    def _index(self) -> int:
        try:
            return find_device(self.sd, self.output_name, "output")
        except SetupError:
            return pick_output(self.sd, self.output_query)

    def _hint(self) -> dict:
        """For HELLO: how long this Mac's output takes to the ear, so the host can make room for it
        before the first audio (AirPods ~0.2 s), and what kind of device it is."""
        try:
            index = self._index()
            name = self.sd.query_devices(index)["name"]
            facts = sacore.mac_call("device_info", name, True)
            dev = self._device(name, (facts or {}).get("uid"))
            plan = ae.plan_output(self.sd, index, 48000, facts, dev.get("io_level", 0))
            lead = dev.get("lead_ms") or ((plan.fixed_s or 0.0) + (plan.blocksize or ae.OUT_BLOCK) / 48000) * 1000
            return {"lead_ms_hint": round(lead, 1), "transport": (facts or {}).get("transport")}
        except Exception:       # only a hint: the first STATS has the measured lead
            return {}

    def _open_output(self, rate: int, channels: int, player: Player, reason: str = "start", level: int | None = None):
        index = self._index()
        self.output_name = self.sd.query_devices(index)["name"]
        out = ae.AudioOutput(self.sd, index, rate, player, self.settings, self._device(self.output_name),
                             log=self.log, reason=reason, level=level)
        self._opened(out, self.client.event)
        return out

    def _tick_loop(self):
        ticks, failed = 0, False
        while not self._stop.wait(self.TICK):
            ticks += 1
            try:
                # (the first time too: PortAudio's idea of the Mac's output may be older than macOS's)
                if ticks == 1 or self.system.changed() or (not self.system.counter and ticks % 4 == 0):
                    self._follow_output()
                if ticks % 2 == 0:
                    c = self.client
                    self._remember(c.output, c.report)
                    act = c.watchdog()
                    if act:
                        self._act(act, c.output, c.event, self._reopen)
                    elif c.output is None and c.player and c.phase == "connected" and ticks % 6 == 0:
                        self._reopen_client(reason="retry")
            except Exception as e:
                if not failed:
                    failed = True
                    self.log(f"! listener tick: {e!r}\n{traceback.format_exc()}")

    def _reopen(self, level, reason, follow):
        if follow:          # the Mac's own output now - unless --output named one, or it's a combined one
            current = self.system.current() if self.system.available and not self.output_query else ""
            if current and not is_combined(current):
                self.output_name = current
            self._reopen_client(before_open=lambda: refresh_portaudio(self.sd), reason=reason)
        else:
            self._reopen_client(level=level, reason=reason)

    def _reopen_client(self, **kw):
        """Client.reopen_output; if the output can't be opened (a Bluetooth device still connecting, say),
        the music goes on without it here and the tick tries again every few seconds."""
        c = self.client
        try:
            c.reopen_output(**kw)
            self._failing = False
        except Exception as e:      # noqa: BLE001 - tried again
            if not self._failing:
                self.log(f"! couldn't play on this Mac ({e}); trying again every few seconds")
                c.event("output_failed", error=str(e))
            self._failing = True

    def _follow_output(self):
        """Put AirPods in (or pick another speaker) and the music moves with you."""
        if self.output_query or not self.system.available:
            return
        current = self.system.current()
        changed, self._seen_output = current != self._seen_output, current
        if not current or not changed or current == self.output_name or is_combined(current):
            return
        self.output_name = current
        self._reopen_client(before_open=lambda: refresh_portaudio(self.sd))
        self.log(f'Now playing on "{self.output_name}".')

    def stop(self):
        self._stop.set()
        if self.client:
            self.client.stop()
        if getattr(self, "system", None):
            self.system.close()

    def info(self) -> dict:
        c = self.client
        out = c.output
        return {
            "mode": "listening", "address": self.address, "host_name": c.host_name, "host_version": c.host_version,
            "connection": c.phase, "output": self.output_name, "error": c.error, "problem": c.problem,
            "delay_ms": round(c.delay * 1000) if c.delay is not None else None,
            "group_volume": c.group_volume, "group_volume_by": c.group_volume_by,
            "clock": c.clock.estimate is not None, **c.stats(),
            "incoming_silent_s": (lambda q: None if q is None else round(min(q, 9999), 1))(
                c.player.silent_for() if c.player else None),
            "audio": {"output": out.info() if out else None},
            "clock_pings": list(c.clock.ping_log)[-60:],
            "events": list(c.events),
        }


# ----------------------------------------------------------------- terminal
def sync_word(p: Player) -> str:
    state = p.state
    if state == "waiting":
        return "waiting"
    if state == "no_audio":
        return f"no audio arriving ({p.dropouts} dropouts)"
    word = f"in sync (±{max(abs(p.err) * 1000, 0.1):.1f} ms)" if state == "in_sync" else \
        f"syncing ({p.err * 1000:+.1f} ms)"
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


class Reporter:
    """--report FILE: a JSON line a second with the session's info() and, when hosting, the stamp of
    every block since the last line ([n, play_at]): what CI's recorder judges the sound against."""

    def __init__(self, path: str, session):
        self.path, self.session = path, session
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="report", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(2)

    def _run(self):
        stamps = getattr(getattr(self.session, "host", None), "stamps", None)
        with open(self.path, "w") as f:
            while not self._stop.wait(1.0):
                line = {"t": time.monotonic(), "info": self.session.info()}
                if stamps is not None:
                    line["stamps"] = [stamps.popleft() for _ in range(len(stamps))]
                f.write(json.dumps(line, default=str) + "\n")
                f.flush()


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


def start_stress(args):
    """--stress (CI): started once the session's gc.freeze() is done, so its 300k-object heap is one a
    full garbage collection still has to walk, as an app's own would be."""
    if not args.stress:
        return None
    import gil_stress
    return gil_stress.Stress(args.stress).start()


def run_host(args):
    sd = import_sounddevice()
    settings = Settings(args.offset, channel=args.channel)
    with Console(None if args.no_local else settings) as con:
        session = HostSession(sd, args.name or computer_name(), args.delay, args.click, not args.no_local,
                              args.input, args.output, settings, args.port, not args.no_switch, log=con.log,
                              opus=args.codec != "pcm", tone=args.tone, stamps=bool(args.report))
        session.start()
        gc.freeze()     # all there is so far stays: a full collection never has to walk it again
        stress = start_stress(args)
        report = Reporter(args.report, session).start() if args.report else None
        try:
            ip = lan_ip()
            delay = "automatic delay" if args.delay is None else f"{args.delay:.0f} ms delay"
            con.log(f'Hosting as "{session.name}" at {ip}  ({session.host.rate} Hz, {delay}, '
                    f"starting at {session.delay * 1000:.0f} ms)")
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
            if report:
                report.stop()
            if stress:
                stress.stop()
            session.stop()


def run_client(args):
    sd = import_sounddevice()
    settings = Settings(args.offset, channel=args.channel)
    with Console(settings) as con:
        session = JoinSession(sd, args.host, args.name or computer_name(), args.output, settings,
                              args.port, log=con.log).start()
        gc.freeze()
        stress = start_stress(args)
        report = Reporter(args.report, session).start() if args.report else None
        con.log(f"Playing on: {session.output_name}")
        con.log(f"Looking for the host at {args.host}..." if args.host else "Looking for the host on this network...")
        if con.tty:
            con.log("  " + con.HELP)
        try:
            while not con.quit.wait(0.25):
                con.status(session.client.status())
        finally:
            if report:
                report.stop()
            if stress:
                stress.stop()
            session.stop()


def run_devices(_args):
    sd = import_sounddevice()
    print(sd.query_devices())
    if sacore.mac is None:
        return
    print("\nmacOS's own figures (frames):")
    for d in sd.query_devices():
        for output in (True, False):
            if d[f"max_{'output' if output else 'input'}_channels"] <= 0:
                continue
            f = sacore.mac_call("device_info", d["name"], output)
            if f:
                print(f"  {d['name']} ({'out' if output else 'in'}): uid {f['uid']} transport {f['transport']} "
                      f"latency {f['latency']} safety {f['safety']} stream latency {f['stream_latency']} "
                      f"buffer {f['buffer_frames']} [{f['buffer_min']}-{f['buffer_max']}] "
                      f"{f['nominal_rate']:.0f} Hz")


VERDICTS = {        # what "Test my headphones" says, in the app and here
    "clean": "Clean: macOS played every bit of audio on these headphones.",
    "macos_dropping": "macOS is dropping audio on these headphones. That static is between macOS and the "
                      "headphones, not in SyncAudio (its part took well under a millisecond). Try: quit apps that "
                      "use the mic (Zoom, FaceTime), turn Bluetooth off and on, unpair other Bluetooth devices "
                      "nearby, and check whether the same song from the Music app also crackles. Then send us "
                      "diagnostics.",
    "app_problem": "SyncAudio itself fell behind on this Mac. Please send us diagnostics.",
}


def run_selftest(args):
    """"Test my headphones" from the command line: the 1.1.6 way, then 2.0's, and a verdict."""
    sd = import_sounddevice()
    index = pick_output(sd, args.output)
    log = (lambda m: print(m, file=sys.stderr)) if args.json else print
    if not args.json:
        print(f'Testing "{sd.query_devices(index)["name"]}": a quiet tone' +
              (", first 10 s the old way (may crackle), then 20 s with SyncAudio 2.0" if not args.no_legacy
               else " for 20 s"))
    result = ae.SelfTest(sd, index, legacy=not args.no_legacy, log=log).run()
    if args.json:
        print(json.dumps(result, default=str))
        return
    if "legacy" in result:
        lg = result["legacy"]
        print(f"old way: buffers of {lg['frames_min']}-{lg['frames_max']} frames, "
              f"{lg['lost_ms_per_s']} ms lost per second")
    v2 = result["v2"]
    print(f"2.0: buffers of {v2['io_frames']} frames, {v2['skips']} skipped cycles ({v2['skipped_ms']} ms), "
          f"{v2['overloads']} overloads, {v2['underruns']} underruns, slowest callback {v2['cb_max_us']} µs")
    print(VERDICTS[result["verdict"]])


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
        # for CI and debugging
        p.add_argument("--stress", choices=["app", "ci"], help=argparse.SUPPRESS)
        p.add_argument("--report", metavar="FILE", help=argparse.SUPPRESS)

    h = sub.add_parser("host", help="share this Mac's audio")
    common(h)
    h.add_argument("--delay", type=float, default=None, metavar="MS",
                   help="fixed capture-to-speaker delay; default: automatic (as low as the Wi-Fi allows)")
    h.add_argument("--input", default="BlackHole", help="device to capture (default: BlackHole)")
    h.add_argument("--click", action="store_true", help="stream a metronome to check the sync by ear")
    h.add_argument("--no-local", action="store_true", help="don't play on this Mac")
    h.add_argument("--no-switch", action="store_true", help="don't change this Mac's sound output")
    h.add_argument("--codec", choices=["auto", "pcm"], default="auto", help=argparse.SUPPRESS)
    h.add_argument("--tone", action="store_true", help=argparse.SUPPRESS)     # with --click: 997 Hz on the left
    h.set_defaults(func=run_host)

    j = sub.add_parser("join", help="play the host's audio on this Mac")
    j.add_argument("host", nargs="?", help="host IP or name.local (default: find it automatically)")
    common(j)
    j.set_defaults(func=run_client)

    sub.add_parser("devices", help="list audio devices").set_defaults(func=run_devices)

    t = sub.add_parser("selftest", help="test an output (e.g. AirPods): the old way, then 2.0's")
    t.add_argument("--output", help="the output to test (default: the Mac's current one)")
    t.add_argument("--no-legacy", action="store_true", help="skip the 10 s with the old way")
    t.add_argument("--json", action="store_true", help="print the result as JSON")
    t.set_defaults(func=run_selftest)

    args = ap.parse_args(argv)
    # No audio thread needs the GIL any more (CoreAudio's threads only run C); this is fairness for the
    # output engine's thread, which should get Python back within a millisecond when it wakes.
    sys.setswitchinterval(0.001)
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
