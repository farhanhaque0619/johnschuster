"""Tests for syncaudio. No audio hardware needed: devices are simulated.

    python3 -m unittest discover tests

Two kinds of simulated devices (tests/vdevice.py):
- virtual time (VirtualSD): the real C callbacks driven by a simulated clock, so a minute of audio
  takes a second and everything is exact and repeatable;
- real time (FakeSoundDevice below): every stream is libsacore's fake IO thread with hard deadlines,
  like CoreAudio's, so whole sessions - several Macs in one process, real sockets - can be played.

What an output played is judged at the speaker, by true time. The fake BlackHole captures its own
frame index k as the sample value ((k mod 32768) - 16384) / 32768 (vdevice.code: exact through the
wire's 16-bit PCM too), and the host's sample index n is that k, so every frame that reaches a speaker
says which captured sample it is; the host's stamps say when that sample was due. Timing is checked
to a fraction of a sample, and continuity frame by frame.
"""
import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
sys.path.insert(0, here)
import sacore  # noqa: E402
import audio_engine as ae  # noqa: E402
import syncaudio as sa  # noqa: E402
import vdevice as vd  # noqa: E402

RATE = 48000
has_fake = sacore.lib is not None and hasattr(sacore.lib, "sa_fake_start")


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_until(check, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.2)
    return False


def hermetic():
    """Session tests never touch this computer's real sound settings: no SwitchAudioSource and no
    macOS helper (on a Mac, sacore.mac would read and switch the real default output)."""
    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch.object(sa.shutil, "which", return_value=None))
    stack.enter_context(mock.patch.object(sacore, "mac", None))
    return stack


class FakeSoundDevice:
    """Just enough of the sounddevice module for whole sessions: BlackHole, a mic and speakers (tests
    add AirPods). Every stream runs libsacore's fake IO thread on the C callback the session gave it
    (vdevice.FakeStream), with that device's profile: lead (IOProc to ear), wobble of its timestamps,
    skip_every (macOS dropping a cycle), and what it records. Outputs keep what they played."""

    __version__ = "0.5.6"
    _libname = None                                     # no real PortAudio to ask for device ids
    _ffi = types.SimpleNamespace(cast=lambda type, x: x)

    class PortAudioError(Exception):
        pass

    DEVICES = [
        {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2, "default_samplerate": 48000.0,
         "default_low_output_latency": 0.002},
        {"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0,
         "default_samplerate": 48000.0},
        {"name": "MacBook Pro Speakers", "max_input_channels": 0, "max_output_channels": 2,
         "default_samplerate": 48000.0, "default_low_output_latency": 0.03},
    ]
    PROFILES = {"MacBook Pro Speakers": {"lead": 0.03},
                "BlackHole 2ch": {"lead": 0.005},
                # Bluetooth: ~0.2 s from the IO cycle to the ear, timestamps that wobble
                "Farhan’s AirPods Pro": {"lead": 0.20, "wobble": 0.001},
                # ...and a Mac that drops an IO cycle every few seconds by itself (Saifan's)
                "Saifan’s AirPods Pro": {"lead": 0.20, "wobble": 0.001, "skip_every": 3.0}}

    def __init__(self, record: float = 30.0):
        self.default = types.SimpleNamespace(device=[0, 0])     # output already on BlackHole
        self.record = record
        self.outputs, self.inputs, self.opened = [], [], []

    def query_devices(self, device=None, kind=None):
        devices = [{"max_output_channels": 0, "max_input_channels": 0, "default_samplerate": 48000.0,
                    "default_low_output_latency": 0.01, "default_high_output_latency": 0.1,
                    "default_low_input_latency": 0.01, "default_high_input_latency": 0.1, **d}
                   for d in self.DEVICES]
        return devices if device is None else devices[device]

    def get_portaudio_version(self):
        return 1246976, "PortAudio V19.7.0-devel (fake)"

    def _StreamBase(self, kind, samplerate, blocksize, device, channels, dtype, latency, callback, userdata,
                    wrap_callback):
        self.opened.append(dict(kind=kind, device=device, blocksize=blocksize, latency=latency,
                                callback=callback, wrap_callback=wrap_callback))
        name = self.DEVICES[device]["name"]
        profile = {"record": self.record if kind == "output" else 0, **self.PROFILES.get(name, {})}
        st = vd.FakeStream(profile, kind, channels, blocksize, callback, userdata)
        st.device, st.name = device, name
        (self.outputs if kind == "output" else self.inputs).append(st)
        return st


def stamps_of(host):
    """The host's stamps as arrays (n, play_at), oldest first."""
    s = np.array(list(host.stamps), dtype=np.float64)
    return s[:, 0], s[:, 1]


def heard(stream, stamps, t_from=-np.inf, t_to=np.inf, settle=0.0):
    """What a closed fake output played - from its first sound (or `settle` s after it: a new output's
    clock takes a few seconds to average out wobbling timestamps) to when it stopped - judged against
    the host's stamps.

    Each recorded frame reached the ear at a known true time. The stamps say which sample n was due
    then: on the line through each block's stamp and the next (exact through glides); in the silence a
    delay increase inserts, nothing. The frame's own value says which sample it is (the fake BlackHole's
    code). Returns err (s, + = late) over the frames that decode cleanly, and the glitches: runs of
    frames that are silent where sound is due, sound where silence is due, or not the next sample."""
    n, at = stamps
    tau, rec = stream.tau(), stream.rec
    st = stream.stats
    played = min(len(rec), (st["cycles"] + st["skipped"] + st["forced"]) * stream.frames)
    first = np.flatnonzero(rec[:played, 1])[0]                          # its first sound (a fade-in)
    sel = np.zeros(len(rec), bool)
    sel[max(0, first - 512 if not settle else first + int(settle * RATE)):played] = True
    sel &= (tau >= max(t_from, at[0] + 0.05)) & (tau < min(t_to, at[-1]))
    before = np.nonzero(sel)[0] < first                                 # silence before it: not judged
    if not sel.any():
        return {"glitches": 0, "glitch_at": [], "err_p99_ms": None, "err_max_ms": None, "err_median_ms": None,
                "good_s": 0.0, "judged_s": 0.0}
    tau, y = tau[sel], rec[sel].astype(np.float64)
    k = np.clip(np.searchsorted(at, tau, "right") - 1, 0, len(at) - 2)
    span = at[k + 1] - at[k]
    straight = np.abs(span - (n[k + 1] - n[k]) / RATE) < 0.001        # the next block is on the same line
    slope = np.where(straight, span / (n[k + 1] - n[k]), 1.0 / RATE)
    due = n[k] + (tau - at[k]) / slope
    gap = ~straight & (due >= n[k + 1])                                 # a delay increase's silence
    # silence: 8 or more zero frames in a row (the code itself is 0 once every P samples)
    zero = ((np.abs(y[:, 0]) < 1e-7) & (np.abs(y[:, 1]) < 1e-7)).astype(int)
    run8 = np.convolve(zero, np.ones(8, int), "valid") == 8
    silent = np.convolve(run8.astype(int), np.ones(8, int))[:len(zero)] > 0
    P = vd.CODE
    m = y[:, 1] * P + P // 2                                            # n mod P, to a fraction of a sample
    dec = due + (m - due + P // 2) % P - P // 2                         # unwrapped near what's due
    # The code jumps back to 0 every P samples, and the resampler rings for 16 samples around that:
    # find those from where the content really is (it may still be a few samples off what's due)
    off = np.zeros(len(tau))
    for i in range(0, len(tau), 1024):
        off[i:i + 1024] = np.median((dec - due)[i:i + 1024])
    wrap = ((due + off) % P < 40) | ((due + off) % P > P - 40)
    # A delay step fades out (5 ms) into its silence and in again: those ends may be silent too, and
    # where its silence begins and ends is only as exact as the timing (frames, not samples)
    near_gap = np.convolve(gap, np.ones(2 * 241 + 1), "same") > 0
    gap_edge = near_gap & (np.convolve(~gap, np.ones(2 * 48 + 1), "same") > 0)
    # fades (at a step, and 64 frames at the edges of a dropout) aren't the clean code: the frames next
    # to silence are judged by the silence itself
    edge = np.convolve(silent | gap, np.ones(2 * 241 + 1), "same") > 0
    good = ~gap & ~silent & ~wrap & ~edge
    jump = np.zeros(len(tau), bool)
    jump[1:] = (np.abs(np.diff(dec) - np.diff(due)) > 0.05) & good[1:] & good[:-1]
    bad = np.where(gap, ~silent & ~gap_edge, (silent & ~near_gap) | jump) & ~before
    good &= ~jump
    where = np.nonzero(bad)[0]
    runs = int(len(where) > 0) + int(np.count_nonzero(np.diff(where) > 480))
    starts = tau[where[np.r_[True, np.diff(where) > 480]]] if len(where) else np.array([])
    err = (due[good] - dec[good]) / RATE
    a = np.abs(err) * 1000
    return {"glitches": runs, "glitch_at": [round(float(t - at[0]), 2) for t in starts[:10]],
            "err_p99_ms": round(float(np.percentile(a, 99)), 4) if len(a) else None,
            "err_max_ms": round(float(a.max()), 4) if len(a) else None,
            "err_median_ms": round(float(np.median(err)) * 1000, 4) if len(a) else None,
            "good_s": round(float(good.sum()) / RATE, 1), "judged_s": round(len(tau) / RATE, 1)}


def missed(*streams) -> int:
    """Cycles the fake devices lost on their own: late ones (a shared machine waking the fake late, or
    pausing it inside a callback that takes microseconds - test_engine_realtime's R1 judges that cost),
    and the ones a profile drops on purpose."""
    return sum(s.stats["late"] + s.stats["forced"] for s in streams if s.stats)


# ---------------------------------------------------------------------- virtual time
SETTINGS = types.SimpleNamespace(offset_ms=0.0, volume=1.0, channel="both")


def speaker(name="Speakers", lead=0.03, **kw):
    return {"name": name, "max_output_channels": 2, "default_low_output_latency": lead, "lead": lead, **kw}


class Mac:
    """A Player and its output on a virtual device, the engine ticking every 10 ms."""

    def __init__(self, sim, device, to_server=lambda t: t, server_rate=lambda: 1.0, local=lambda t: t,
                 settings=None):
        self.sim, self.local = sim, local
        self.sd = vd.VirtualSD(sim, [{**device, "local": local}])
        self.settings = settings or sa.Settings()
        self.player = sa.Player(RATE, 2, to_server, self.settings, server_rate, now=lambda: local(sim.t))
        self.open()

    def open(self):
        self.out = ae.AudioOutput(self.sd, 0, RATE, self.player, self.settings, {}, log=lambda m: None,
                                  now=lambda: self.local(self.sim.t), threads=False)
        self.sim.every(ae.OutputEngine.TICK, self.out.engine.tick, skip=lambda t, o=self.out: not o.stream.active)

    @property
    def dev(self):
        return self.out.stream.device

    def judge(self, due, t_from, t_to=np.inf, allow=()):
        return vd.judge(self.dev.tape, self.dev.tau(), due, t_from, t_to, allow)

    def close(self):
        self.out.close()
        self.player.close()


class PlayerTest(unittest.TestCase):
    """The Player facade over the engine, on one virtual Mac."""

    D = 0.45

    def setUp(self):
        self.sim = vd.Sim()
        self.T0 = self.sim.t + 0.1
        self.macs = []

    def tearDown(self):
        for m in self.macs:
            m.close()

    def mac(self, **kw):
        m = Mac(self.sim, speaker(**kw.pop("device", {})), **kw)
        self.macs.append(m)
        return m

    def feed(self, player, until=np.inf, stalls=(), skip=()):
        """The host's blocks, each arriving 5 ms after it was captured (later in a stall), until the host
        stops; blocks whose index is in `skip` never arrive."""
        def feed(n, at, pcm):
            if n not in skip and n < until * RATE:
                player.feed(n, at, pcm)
        return vd.Feeder(self.sim, types.SimpleNamespace(feed=feed), self.T0, delay=lambda n: self.D, stalls=stalls)

    def test_plays_each_sample_at_its_time_and_says_so(self):
        m = self.mac()
        states = []
        self.sim.every(0.1, lambda: states.append(m.player.state), first=self.sim.t)
        self.feed(m.player)
        self.sim.run(10)
        self.assertEqual(states[0], "waiting")
        self.assertEqual(states[-1], "in_sync")
        first = states.index("in_sync")
        self.assertTrue(all(s == "in_sync" for s in states[first:]), states)
        self.assertLess(first * 0.1, self.D + 1.5)          # in sync a second after the music starts
        r = m.judge(vd.steady(self.T0, self.D), self.T0 + self.D + 0.1, self.T0 + 9.5)
        self.assertEqual(r["glitches"], 0, r)
        self.assertLess(r["err_max_ms"], 0.01)
        self.assertEqual((m.player.dropouts, m.player.resyncs, m.player.skips), (0, 0, 0))
        self.assertLess(abs(m.player.err), 1e-5)

    def test_running_dry_is_a_dropout_and_no_audio_then_exact_again(self):
        m = self.mac()
        states = {}
        self.sim.every(0.05, lambda: states.setdefault(round(self.sim.t - self.T0, 2), m.player.state))
        self.feed(m.player, stalls=[(self.T0 + 4.0, 0.8)])         # Wi-Fi stalls for longer than the delay
        self.sim.run(10)
        self.assertIn("no_audio", [s for t, s in states.items() if 4.0 < t < 5.5])
        self.assertEqual(states[max(states)], "in_sync")
        self.assertEqual(m.player.dropouts, 1)
        self.assertEqual(m.player.resyncs, 1)               # the silent re-entry after running dry
        r = m.judge(vd.steady(self.T0, self.D), self.T0 + 5.6, self.T0 + 9.5)
        self.assertEqual(r["glitches"], 0, r)
        self.assertLess(r["err_max_ms"], 0.01)              # the gap was silence, not lag

    def test_audio_that_never_arrives_is_a_dropout(self):
        m = self.mac()
        self.feed(m.player, skip={480 * 300})            # one block lost (the host let it go: too late)
        self.sim.run(6)
        self.assertEqual(m.player.hole_events, 1)
        self.assertEqual(m.player.dropouts, 1)
        self.assertEqual(m.player.sb.holes, 480)
        self.assertEqual(m.player.second(m.out)["holes"], 480)

    def test_offset_plays_later(self):
        settings = sa.Settings(offset_ms=10)
        m = self.mac(settings=settings)
        self.feed(m.player)
        self.sim.run(6)
        r = m.judge(vd.steady(self.T0, self.D + 0.010), self.T0 + self.D + 0.2, self.T0 + 5.5)
        self.assertEqual(r["glitches"], 0, r)
        self.assertLess(r["err_max_ms"], 0.01)

    def test_counts_carry_on_across_a_reopened_output(self):
        m = self.mac()
        self.feed(m.player, stalls=[(self.T0 + 3.0, 0.8)])
        self.sim.run(5)
        self.assertEqual(m.player.dropouts, 1)
        m.out.close()                                       # e.g. a bigger buffer, or AirPods put in
        m.open()
        self.sim.run(12)
        self.assertEqual(m.player.dropouts, 1)
        rep = m.player.second(m.out)
        self.assertEqual((rep["underruns"], rep["dropouts"]), (1, 1))
        self.assertEqual(m.player.state, "in_sync")
        r = m.judge(vd.steady(self.T0, self.D), self.sim.t - 4, self.sim.t - 0.5)
        self.assertEqual(r["glitches"], 0, r)
        self.assertLess(r["err_max_ms"], 0.01)

    def test_a_restarted_host_is_played_from_its_new_start(self):
        m = self.mac()
        self.feed(m.player, until=3.5)
        self.sim.run(4)
        later = self.sim.t + 1.0
        for k in range(300):                                # a new host stream: n starts at 0 again
            self.sim.at(later + k * 0.01, lambda k=k: m.player.feed(k * 480, later + 0.6 + k * 0.01,
                                                                    vd.signal(k * 480, 480)))
        self.sim.run(8)
        r = m.judge(vd.steady(later + 0.6, 0.0), later + 0.7, later + 3.0)
        self.assertEqual(r["glitches"], 0, r)
        self.assertLess(r["err_max_ms"], 0.01)

    def test_a_closed_player_ignores_late_audio(self):
        m = self.mac()
        self.macs.remove(m)
        m.close()
        m.player.feed(0, 1.0, vd.signal(0, 480))            # a network thread that hadn't noticed yet
        self.assertEqual(m.player.state, "waiting")
        self.assertIsNone(m.player.silent_for())


class SettingsTest(unittest.TestCase):
    def test_each_output_keeps_its_own_nudge(self):
        s = sa.Settings(offset_ms=5)
        self.assertEqual(s.offset_ms, 5)
        s.device_key = "airpods-uid"
        self.assertEqual(s.offset_ms, 5)                    # nothing of its own yet
        s.offset_ms = 40
        s.device_key = "speakers-uid"
        self.assertEqual(s.offset_ms, 5)
        s.offset_ms -= 2
        s.device_key = "airpods-uid"
        self.assertEqual(s.offset_ms, 40)
        self.assertEqual(s.offsets, {"airpods-uid": 40, "speakers-uid": 3})


class PlayerSimulationTest(unittest.TestCase):
    """Two Macs in virtual time: the host plays on its speakers, a friend on AirPods. The friend's clock
    is offset and runs 80 ppm fast; each sound card runs off its own crystal; the Wi-Fi is jittery and
    lopsided, so ClockSync's view of the host's clock wanders. Judged at both speakers by true time."""

    D = 0.59

    def simulate(self, seconds=40.0, stall=None, seed=1, clock_ppm=80.0, host_ppm=-35.0, friend_ppm=60.0):
        rng = np.random.default_rng(seed)
        sim = vd.Sim()
        T0 = sim.t + 0.2
        k = clock_ppm * 1e-6
        local = lambda T: 777.0 + T * (1 + k)                           # noqa: E731 - the friend's clock
        clock = sa.ClockSync()
        host = Mac(sim, speaker(lead=0.025, ppm=host_ppm, seconds=seconds))
        friend = Mac(sim, speaker("AirPods", lead=0.2, ppm=friend_ppm, seconds=seconds,
                                  report=lambda i: float(rng.uniform(-0.001, 0.001))),
                     clock.to_server, clock.server_rate, local)
        last = [0.0]

        def capture(i):
            n = i * vd.BLOCK
            block = (n, T0 + n / RATE + self.D, vd.signal(n, vd.BLOCK))
            host.player.feed(*block)
            arrive = sim.t + 0.002 + rng.exponential(0.004)
            if stall and T0 + stall <= sim.t < T0 + stall + 0.6:
                arrive = T0 + stall + 0.6 + 0.005                       # Wi-Fi hiccup: everything waits
            last[0] = max(last[0], arrive)                              # TCP keeps the order
            sim.at(last[0], lambda: friend.player.feed(*block))
            sim.at(T0 + (i + 2) * vd.BLOCK / RATE, lambda: capture(i + 1))

        def ping():
            up, down = 0.0015 + rng.exponential(0.003), 0.001 + rng.exponential(0.002)
            t2 = sim.t + up
            clock.add(local(sim.t), t2, t2 + 1e-4, local(t2 + 1e-4 + down))
            sim.at(sim.t + (0.03 if sim.t - T0 < 2 else 0.2), ping)
        sim.at(T0 + vd.BLOCK / RATE, lambda: capture(0))
        sim.at(sim.t, ping)
        sim.run(seconds)
        due = vd.steady(T0, self.D)
        out = {name: (m, m.judge(due, T0 + 10, T0 + seconds - 1)) for name, m in (("host", host), ("friend", friend))}
        for m in (host, friend):
            m.close()
        return out, clock

    def test_two_macs_stay_in_sync_despite_drift(self):
        res, clock = self.simulate()
        (host, h), (friend, f) = res["host"], res["friend"]
        print(f"\n  simulated worst error after 10 s: host {h['err_max_ms']:.3f} ms, friend {f['err_max_ms']:.3f} ms",
              file=sys.stderr)
        self.assertEqual((h["glitches"], f["glitches"]), (0, 0), (h, f))
        self.assertLess(h["err_max_ms"], 0.01)
        self.assertLess(f["err_max_ms"], 1.0)
        self.assertEqual((host.player.resyncs, friend.player.resyncs), (0, 0))
        self.assertEqual((host.player.dropouts, friend.player.dropouts), (0, 0))
        self.assertAlmostEqual(clock.server_rate(), 1 / (1 + 80e-6), delta=20e-6)

    def test_extreme_drift(self):
        res, _ = self.simulate(clock_ppm=-400, host_ppm=300, friend_ppm=-300, seed=7)
        (_, h), (_, f) = res["host"], res["friend"]
        self.assertEqual((h["glitches"], f["glitches"]), (0, 0), (h, f))
        self.assertLess(h["err_max_ms"], 0.01)
        self.assertLess(f["err_max_ms"], 1.0)

    def test_a_wifi_stall_is_a_gap_not_lag(self):
        res, _ = self.simulate(stall=20.0, seed=2)
        friend, f = res["friend"]
        self.assertGreaterEqual(friend.player.dropouts, 1)      # 0.6 s: more than the AirPods leave
        self.assertLessEqual(f["glitches"], 1, f)               # one gap...
        self.assertLess(f["err_max_ms"], 1.0)                   # ...and every sample still on time
        self.assertEqual(friend.player.total("splices"), 0)


class ClickSourceTest(unittest.TestCase):
    def test_blocks_are_stamped_on_one_line_and_tone_is_on_the_left(self):
        blocks = []
        clicks = sa.ClickSource(RATE, 2, lambda pcm, n, t: blocks.append((n, t, pcm.copy())), tone=True)
        clicks.start()
        time.sleep(1.2)
        clicks.close()
        n = np.array([b[0] for b in blocks])
        t = np.array([b[1] for b in blocks])
        self.assertTrue(np.all(np.diff(n) == 480))
        self.assertLess(np.abs(t - (t[0] + (n - n[0]) / RATE)).max(), 1e-9)
        left = np.concatenate([b[2][:, 0] for b in blocks])
        right = np.concatenate([b[2][:, 1] for b in blocks])
        self.assertAlmostEqual(np.abs(left).max(), 10 ** (-12 / 20), places=3)
        expect = sa.ClickSource.TONE_LEVEL * np.sin(2 * np.pi * 997 * np.arange(len(left)) / RATE)
        np.testing.assert_allclose(left, expect, atol=1e-5)
        self.assertGreater(np.abs(right).max(), 0.5)            # the clicks
        self.assertGreater(np.mean(right == 0), 0.9)            # with silence between them


class GroupVolumeTest(unittest.TestCase):
    def test_changes_glide_over_one_block(self):
        host = sa.Host(RATE, 1, 0.1, free_port(), log=lambda *a: None)
        host.local = sa.Player(RATE, 1, lambda t: t)
        ones = np.ones((240, 1), np.float32)
        host.on_block(ones, 0, 10.0)
        host.group_volume = 0.5
        host.on_block(ones, 240, 10.0 + 240 / RATE)
        host.on_block(ones, 480, 10.0 + 480 / RATE)
        played = np.zeros((720, 1), np.float32)
        host.local.sb.content.resample(0.0, 1.0, 720, played)   # speed 1 on whole samples: exactly what's held
        played = played[:, 0]
        np.testing.assert_allclose(played[:240], 1.0)
        self.assertTrue(np.all(np.diff(played[240:480]) <= 0) and played[240] == 1.0 and played[479] == 0.5)
        np.testing.assert_allclose(played[480:], 0.5)
        host.local.close()


class SlowWifiTest(unittest.TestCase):
    """Saifan: "it stays for a minute or two, then really bad static, more delay
    every few seconds, then it gets dropped." A listener whose Wi-Fi can't keep
    up must get only audio it can still play, not an ever-growing backlog."""

    def test_audio_that_would_arrive_late_is_never_sent(self):
        peer = sa._Peer(None, "Saifan", "10.49.64.113")
        for i in range(10):
            peer.push(b"a%d" % i, play_at=100.0 + i)
        peer.push(b"update")                        # control messages always go
        self.assertEqual(peer.take(now=106.5), b"a7a8a9update")
        self.assertEqual(peer.late, 7)

    def test_a_stalled_listener_is_not_disconnected(self):
        peer = sa._Peer(None, "Saifan", "10.49.64.113")
        for i in range(sa._Peer.QUEUE + 50):        # Wi-Fi stalled for 3.5 s
            peer.push(b"x", play_at=float(i))
        self.assertTrue(peer.alive)
        self.assertEqual(peer.q.qsize(), sa._Peer.QUEUE)
        self.assertEqual(peer.q.queue[0][0], 50.0, "the oldest audio is the one let go")
        self.assertEqual(peer.late, 50)


class SystemOutputTest(unittest.TestCase):
    """The Mac's sound output: through libsacore's macOS helper when present, else SwitchAudioSource."""

    def fake_mac(self, devices, default):
        state = {"default": default, "set": []}
        counter = types.SimpleNamespace(value=0, close=mock.Mock())
        by_name = {name: i for i, name in devices.items()}
        mac = types.SimpleNamespace(
            SEL={"default_output": 1, "global": 2}, system=1,
            default_output=lambda: state["default"],
            device_info_id=lambda i, out: {"id": i, "name": devices[i], "uid": f"uid{i}"} if i in devices else None,
            device_info=lambda name, out: {"id": by_name[name], "name": name} if name in by_name else None,
            set_default_output=lambda i: state["set"].append(i) or state.update(default=i) or True,
            listen=lambda obj, sel, scope: counter)
        return mac, state, counter

    def test_the_helper_reads_switches_and_notices_changes(self):
        mac, state, counter = self.fake_mac({40: "MacBook Pro Speakers", 41: "BlackHole 2ch", 42: "AirPods"}, 40)
        with mock.patch.object(sacore, "mac", mac), mock.patch.object(sa.subprocess, "run") as run:
            system = sa.SystemOutput(lambda m: None, tool="/opt/bin/SwitchAudioSource")
            self.assertEqual(system.current(), "MacBook Pro Speakers")
            system.switch_to("BlackHole 2ch", None)
            self.assertEqual((state["set"], system.previous), ([41], "MacBook Pro Speakers"))
            self.assertFalse(system.changed())
            counter.value += 1                              # macOS: the default output changed
            self.assertTrue(system.changed())
            self.assertFalse(system.changed())
            system.restore()
            self.assertEqual(state["default"], 40)
            run.assert_not_called()                         # no processes at all
            system.close()
            counter.close.assert_called_once()

    def test_without_the_helper_switchaudiosource_is_spawned_not_forked(self):
        calls = []

        def run(cmd, **kw):
            calls.append((cmd, kw))
            return types.SimpleNamespace(stdout="MacBook Pro Speakers\n")
        with mock.patch.object(sacore, "mac", None), mock.patch.object(sa.subprocess, "run", run):
            system = sa.SystemOutput(lambda m: None, tool="/opt/homebrew/bin/SwitchAudioSource")
            self.assertFalse(system.changed())
            self.assertEqual(system.current(), "MacBook Pro Speakers")
            system.set("BlackHole 2ch")
        self.assertEqual(calls[1][0], ["/opt/homebrew/bin/SwitchAudioSource", "-s", "BlackHole 2ch", "-t", "output"])
        for cmd, kw in calls:
            # CPython 3.12 uses posix_spawn (no fork of the whole app) only with these
            self.assertTrue(os.path.isabs(cmd[0]))
            self.assertIs(kw["close_fds"], False)

    def test_stop_never_leaves_the_mac_on_blackhole(self):
        # the output it had before sharing is gone (AirPods back in their case): where it plays instead
        mac, state, counter = self.fake_mac({40: "MacBook Pro Speakers", 41: "BlackHole 2ch", 42: "AirPods"}, 42)
        with mock.patch.object(sacore, "mac", mac):
            system = sa.SystemOutput(lambda m: None)
            system.switch_to("BlackHole 2ch", None)
            self.assertEqual(system.previous, "AirPods")
            gone = mac.device_info
            mac.device_info = lambda name, out: None if name == "AirPods" else gone(name, out)
            system.restore("MacBook Pro Speakers")
            self.assertEqual(state["default"], 40)
            system.previous = "MacBook Pro Speakers"
            system.restore("Other")                         # (the one before is there: that one)
            self.assertEqual(state["default"], 40)

    def test_a_failing_helper_falls_back(self):
        mac, state, counter = self.fake_mac({40: "MacBook Pro Speakers"}, 40)
        mac.default_output = mock.Mock(side_effect=OSError("HAL"))
        with mock.patch.object(sacore, "mac", mac), mock.patch.object(sacore, "_mac_failed", set()), \
                contextlib.redirect_stderr(io.StringIO()):
            system = sa.SystemOutput(lambda m: None, tool=None)
            self.assertFalse(system.available)
            self.assertEqual(system.current(), "")


# ------------------------------------------------------------------------- real time
AIRPODS = {"name": "Farhan’s AirPods Pro", "max_input_channels": 1, "max_output_channels": 2,
           "default_samplerate": 48000.0, "default_low_output_latency": 0.165}


def quiet(*_):
    pass


def with_airpods(name=AIRPODS["name"], record=30.0):
    sd = FakeSoundDevice(record)
    sd.DEVICES = sd.DEVICES + [{**AIRPODS, "name": name}]
    return sd


def onsets(stream, threshold=0.05):
    """True ear times of the clicks a closed fake output played: the first frame above the threshold
    after 0.1 s of quiet."""
    x, tau = np.abs(stream.rec[:, 1]), stream.tau()
    loud = np.flatnonzero(x > threshold)
    first = loud[np.r_[True, np.diff(loud) > 0.1 * RATE]] if len(loud) else loud
    return tau[first]


@unittest.skipUnless(has_fake, "needs the test build of libsacore")
class EndToEndTest(unittest.TestCase):
    def test_host_and_friend_click_together(self):
        """Compressed with Opus (when this machine has libopus) and as plain samples."""
        codecs = (["opus"] if sa.codec.available() else []) + ["pcm"]
        for name in codecs:
            with self.subTest(codec=name), hermetic(), \
                    mock.patch.object(sa.codec, "available", side_effect=lambda *a: name == "opus"):
                self.click_together(name)

    def click_together(self, codec_name):
        port, D = free_port(), 0.45
        host = sa.Host(RATE, 2, D, port, "test-host", log=quiet, stamps=True)
        host.start()
        hsd, fsd = FakeSoundDevice(15), FakeSoundDevice(15)
        host.local = sa.Player(RATE, 2, lambda t: t)
        host_out = ae.AudioOutput(hsd, 2, RATE, host.local, sa.Settings(), {}, log=quiet)

        def open_output(rate, channels, player, reason="start", level=None):
            return ae.AudioOutput(fsd, 2, rate, player, client.settings, {}, log=quiet, reason=reason, level=level)
        client = sa.Client("127.0.0.1", open_output, port, "test-friend", log=quiet)
        clicks = sa.ClickSource(RATE, 2, host.on_block)
        clicks.start()
        client.start()
        try:
            time.sleep(10)
            listeners = host.listeners()
        finally:
            clicks.close()
            client.stop()
            host_out.close()
            host.local.close()
            host.stop()
        self.assertEqual([(l["name"], l["state"]) for l in listeners], [("test-friend", "in_sync")])
        self.assertEqual(listeners[0]["codec"], codec_name)
        self.assertEqual(client.state, "playing")
        start = host.stamps[0][1]                       # when the first sample was due: the first click
        h, f = onsets(hsd.outputs[0]), onsets(fsd.outputs[-1])
        h, f = h[(h > start + 3.25) & (h < f[-1] + 0.25)], f[f > start + 3]   # clock sync settled; both playing
        self.assertGreater(len(h), 8)
        self.assertGreater(len(f), 8)
        # Each click comes out of the host speaker when it was meant to (its 2nd sample is the first loud one)...
        intended = start + np.round((h - start) * 2) / 2 + 1 / RATE
        self.assertLess(np.abs(h - intended).max(), 0.0001)
        # ...and out of the friend's speaker at the same moment.
        diffs = np.array([f[np.argmin(np.abs(f - t))] - t for t in h])
        print(f"\n  end-to-end host vs friend ({codec_name}): median {np.median(np.abs(diffs)) * 1000:.3f} ms, "
              f"worst {np.abs(diffs).max() * 1000:.3f} ms", file=sys.stderr)
        if codec_name == "pcm":
            self.assertLess(np.abs(diffs).max(), 0.0003)
        else:
            # Like any music codec Opus smears a sharp click a little ahead of itself ("pre-echo"), which
            # now and then trips the click detector early, so judge the typical click.
            self.assertLess(np.median(np.abs(diffs)), 0.0012)
            self.assertGreaterEqual(np.mean(np.abs(diffs) < 0.0012), 0.75)
            self.assertLess(np.abs(diffs).max(), 0.008)

    def test_discovery(self):
        port = free_port()
        host = sa.Host(RATE, 2, 0.1, port, "Living Room Mac", log=lambda *a: None)
        host.start()
        try:
            self.assertEqual(sa.discover(port, address="127.0.0.1"), ("127.0.0.1", "Living Room Mac"))
        finally:
            host.stop()


def played_the_capture(stream, seconds=2.0) -> float:
    """Share of the last `seconds` a closed fake output played that is the fake BlackHole's code, sample
    after sample (what was captured, in order)."""
    y = stream.rec[:, 1].astype(np.float64)
    end = np.flatnonzero(y)[-1]                     # where the music stopped (the session ended)
    step = np.diff(y[end - int(seconds * RATE):end]) * vd.CODE
    return float(np.mean(np.abs(step - 1) < 0.05))


@unittest.skipUnless(has_fake, "needs the test build of libsacore")
class CommandLineTest(unittest.TestCase):
    def setUp(self):
        self.switch = sys.getswitchinterval()       # main() sets it

    def tearDown(self):
        sys.setswitchinterval(self.switch)

    def test_host_and_join_commands(self):
        hosting, joining = FakeSoundDevice(), FakeSoundDevice()
        port = free_port()
        quit_after = lambda seconds: lambda self: (threading.Timer(seconds, self.quit.set).start(), self)[1]  # noqa: E731
        out = io.StringIO()
        report = os.path.join(tempfile.mkdtemp(), "host.jsonl")
        host = threading.Thread(target=sa.main, args=(["host", "--port", str(port), "--name", "Den", "--codec", "pcm",
                                                      "--report", report],))
        # Each "Mac" gets its own fake devices, however late its thread gets going.
        devices = lambda: hosting if threading.current_thread() is host else joining  # noqa: E731
        with mock.patch.object(sa.Console, "__enter__", quit_after(9)), hermetic(), \
                mock.patch.object(sa, "import_sounddevice", side_effect=devices), \
                contextlib.redirect_stdout(out):
            host.start()
            time.sleep(1)
            sa.main(["join", "127.0.0.1", "--port", str(port), "--name", "Sam"])
            host.join(10)
        log = out.getvalue()
        self.assertIn('Set this Mac\'s sound output to "BlackHole 2ch"', log)
        self.assertIn("playing on: MacBook Pro Speakers", log)    # not BlackHole: no feedback loop
        self.assertIn("starting at 450 ms", log)
        self.assertIn("+ Sam joined", log)
        self.assertIn("connected to Den", log)
        for sd in (hosting, joining):
            # (less what the fake devices dropped by themselves - a shared machine waking them late: the
            # capture's cycles are missing on every output, the output's own on that one)
            fakes = [sd.outputs[0], *hosting.inputs]
            lost = sum((s.stats["late"] + s.stats["skipped"]) * s.frames for s in fakes if s.stats) / (2 * RATE)
            self.assertGreater(played_the_capture(sd.outputs[0]), 0.95 - lost, f"what BlackHole captured, in order "
                               f"(fakes: {[s.stats for s in fakes]})")
            for s in sd.opened:                         # I1: PortAudio only ever gets the C callback
                self.assertIsInstance(s["callback"], int)
                self.assertIsNone(s["wrap_callback"])
                self.assertEqual((s["blocksize"], s["latency"]), (512 if s["kind"] == "input" else 1024, 0.0))
        # --report: a line a second for CI, with every block's stamp
        lines = [json.loads(line) for line in open(report)]
        self.assertGreater(len(lines), 5)
        self.assertEqual(lines[2]["info"]["mode"], "hosting")
        stamps = np.array([st for line in lines for st in line["stamps"]])
        self.assertTrue(np.all(np.diff(stamps[:, 0]) == 480))
        self.assertLess(np.abs(np.diff(stamps[:, 1]) - 0.01).max(), 1e-6)    # one line: no delay change

    def test_selftest_command(self):
        sd = vd.FakeSD([{"name": "AirPods", "max_output_channels": 2, "default_low_output_latency": 0.165,
                         "lead": 0.16, "frames": 64}])
        out = io.StringIO()
        with mock.patch.object(sa, "import_sounddevice", return_value=sd), hermetic(), \
                mock.patch.object(ae.SelfTest, "SECONDS", (0.6, 2.5)), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(io.StringIO()):
            sa.main(["selftest", "--output", "AirPods", "--json"])
        result = json.loads(out.getvalue())
        self.assertEqual(result["legacy"]["frames_max"], 64)      # the old way: PortAudio's tiny buffer
        self.assertEqual(result["v2"]["io_frames"], 2048)
        self.assertIn(result["verdict"], ("clean", "macos_dropping"))     # (a shared machine may drop one)
        self.assertIn(result["verdict"], sa.VERDICTS)

    def test_devices_command_shows_macos_figures(self):
        mac = types.SimpleNamespace(device_info=lambda name, out: {
            "uid": f"{name}-uid", "transport": "blue" if "AirPods" in name else "bltn", "latency": 0, "safety": 3840,
            "stream_latency": 3840, "buffer_frames": 2048, "buffer_min": 15, "buffer_max": 4096,
            "nominal_rate": 48000.0} if out else None)
        sd = with_airpods()
        out = io.StringIO()
        with mock.patch.object(sa, "import_sounddevice", return_value=sd), mock.patch.object(sacore, "mac", mac), \
                contextlib.redirect_stdout(out):
            sa.main(["devices"])
        self.assertIn("Farhan’s AirPods Pro (out): uid Farhan’s AirPods Pro-uid transport blue latency 0 safety 3840 "
                      "stream latency 3840 buffer 2048 [15-4096] 48000 Hz", out.getvalue())


@unittest.skipUnless(has_fake, "needs the test build of libsacore")
class SessionTest(unittest.TestCase):
    """HostSession + JoinSession, the pieces the app is built from."""

    def test_three_macs_a_delay_set_by_hand_and_everyones_volume(self):
        port = free_port()
        host_sd = FakeSoundDevice(40)
        with hermetic():
            host = sa.HostSession(host_sd, "Den", port=port, log=quiet, opus=False).start()
            listeners = [sa.JoinSession(FakeSoundDevice(40), "127.0.0.1", name, port=port, log=quiet).start()
                         for name in ("Sam", "Alex")]
            try:
                def everyone():
                    info = host.info()
                    return [l["state"] for l in info["listeners"]] + [info["local"]["state"]]
                self.assertTrue(wait_until(lambda: everyone() == ["in_sync"] * 3, 15), host.info())
                info = host.info()
                self.assertEqual([l["name"] for l in info["listeners"]], ["Alex", "Sam"])
                self.assertTrue(info["auto_delay"])
                self.assertEqual(info["delay_ms"], 450)                 # speakers everywhere: the shortest start
                self.assertEqual(info["capture"]["name"], "BlackHole 2ch")
                self.assertEqual(info["output"], "MacBook Pro Speakers")

                host.set_delay(700)                                     # no quiet moment: a step after 3 s
                self.assertEqual(host.info()["delay_target_ms"], 700)
                self.assertTrue(wait_until(lambda: all(l.info()["delay_ms"] == 700 for l in listeners), 6))
                # 700 ms less the speaker's lead and Wi-Fi (as of each output engine's next 10 ms tick)
                self.assertTrue(wait_until(lambda: all(l.info()["headroom_ms"] > 500 for l in listeners), 3),
                                [l.info()["headroom_ms"] for l in listeners])
                self.assertTrue(wait_until(lambda: everyone() == ["in_sync"] * 3, 6), host.info())
                for l in listeners:
                    state = l.info()
                    self.assertEqual(state["connection"], "connected")
                    self.assertEqual(state["glitches"], 0, state)
                info = host.info()
                self.assertEqual(info["delay_ms"], 700)
                self.assertFalse(info["auto_delay"])
                self.assertEqual([(e["mode"], e["to_ms"]) for e in info["events"] if e["event"] == "delay_change"],
                                 [("manual", 700.0)])
                for l in listeners:         # and each listener's own diagnostics have it
                    self.assertEqual([(e["from_ms"], e["to_ms"], e["mode"]) for e in l.info()["events"]
                                      if e["event"] == "delay_change"], [(450.0, 700.0, "manual")])

                # Sam turns everyone down to half.
                sam, alex = listeners
                sam.client.set_group_volume(0.5)
                time.sleep(2)
                self.assertEqual((host.info()["group_volume"], host.info()["group_volume_by"]), (0.5, "Sam"))
                self.assertEqual((alex.info()["group_volume"], alex.info()["group_volume_by"]), (0.5, "Sam"))
                for sd in [host_sd] + [l.sd for l in listeners]:
                    # the code covers -0.5..0.5 every 0.68 s: at half volume, a quarter
                    level = np.percentile(np.abs(sd.outputs[-1].live(0.7)), 99)
                    self.assertAlmostEqual(level, 0.25, delta=0.01)
            finally:
                for l in listeners:
                    l.stop()
                host.stop()

    def test_a_stall_of_the_sharing_macs_capture_is_no_reason_for_a_longer_delay(self):
        # BlackHole stops calling back; the watchdog reopens it after a second. Everyone hears the same
        # gap, and the audio after it is as early as ever: the delay stays (no urgent step, no raise)
        port = free_port()
        hsd = FakeSoundDevice(40)
        with hermetic():
            host = sa.HostSession(hsd, "Den", port=port, log=quiet, opus=False).start()
            join = sa.JoinSession(FakeSoundDevice(40), "127.0.0.1", "Sam", port=port, log=quiet).start()
            try:
                self.assertTrue(wait_until(lambda: join.info()["state"] == "in_sync", 15))
                time.sleep(6)                               # past the policy's first settle
                blackhole = hsd.inputs[-1]
                blackhole.abort()                           # no more callbacks, though PortAudio says active
                blackhole.active = True
                self.assertTrue(wait_until(lambda: len(hsd.inputs) > 1, 6))
                time.sleep(12)                              # (a raise needs 10 reports)
                changes = [e for e in host.info()["events"] if e["event"] == "delay_change"]
                self.assertEqual(changes, [])
                self.assertGreater(join.info()["dropouts"], 0)                  # (the gap did reach it)
            finally:
                join.stop()
                host.stop()

    def test_missing_blackhole_is_a_setup_error(self):
        sd = FakeSoundDevice()
        sd.DEVICES = [d for d in sd.DEVICES if "BlackHole" not in d["name"]]
        with self.assertRaises(sa.SetupError) as caught:
            sa.HostSession(sd, "Den", port=free_port(), log=lambda *a: None).start()
        self.assertEqual(caught.exception.code, "no_blackhole")

    def test_sharing_can_restart_on_the_same_port(self):
        port = free_port()
        for _ in range(3):
            host = sa.Host(RATE, 2, 0.1, port, "Den", log=lambda *a: None)
            host.start()
            host.stop()

    def test_discover_all(self):
        port = free_port()
        host = sa.Host(RATE, 2, 0.1, port, "Kitchen", log=lambda *a: None)
        host.start()
        try:
            self.assertEqual(sa.discover_all(port, 0.5, ["127.0.0.1", "127.0.0.1"]), [("127.0.0.1", "Kitchen")])
            self.assertEqual(sa.discover_all(free_port(), 0.3, ["127.0.0.1"]), [])
        finally:
            host.stop()


@unittest.skipUnless(has_fake, "needs the test build of libsacore")
class BluetoothAudioTest(unittest.TestCase):
    """AirPods-like outputs (~0.2 s to the ear, wobbling timestamps), played for real on the fake IO
    thread and judged at the "ear" against the host's stamps: every frame on time, in order."""

    def setUp(self):
        self.switch = sys.getswitchinterval()
        sys.setswitchinterval(0.001)            # what the app sets
        self.stack = hermetic()
        self.stack.__enter__()

    def tearDown(self):
        self.stack.close()
        sys.setswitchinterval(self.switch)

    def test_sharing_mac_on_airpods_starts_with_room_for_them_and_plays_exactly(self):
        sd = with_airpods()
        host = sa.HostSession(sd, "Den", port=free_port(), output_query="AirPods", log=quiet, opus=False,
                              stamps=True).start()
        try:
            self.assertEqual(host.info()["delay_ms"], 590)          # their lead, capture, Wi-Fi, a stall, margin
            self.assertTrue(wait_until(lambda: host.info()["local"]["state"] == "in_sync", 8), host.info())
            time.sleep(8)
            info = host.info()
        finally:
            host.stop()
        out, inp = sd.outputs[-1], sd.inputs[-1]
        r = heard(out, stamps_of(host.host), t_from=host.host.stamps[0][1] + 4.0)     # the clock has settled
        print(f"\n  sharing Mac on AirPods: {r}, fake {out.stats}", file=sys.stderr)
        self.assertEqual(sd.opened[0]["blocksize"], 2048)           # Bluetooth: a big IO buffer
        self.assertEqual(info["local"]["io"], [2048, 2048])
        self.assertEqual((info["local"]["underruns"], info["delay_ms"]), (0, 590))
        self.assertLessEqual(info["local"]["skips"], out.stats["late"])         # only the fake's own late wake-ups
        self.assertLessEqual(r["glitches"], missed(out, inp), r)
        self.assertLess(r["err_p99_ms"], 0.75, r)                  # with 1 ms of wobble in every timestamp
        # (a line through a few seconds of such stamps is good to ~0.1 ms, 1 sigma: up to 0.3 ms seen)
        self.assertLess(abs(r["err_median_ms"]), 0.5, r)
        self.assertGreater(r["good_s"], 3)

    def listen(self, airpods: str, seconds: float, stress=None):
        """A host on speakers and a listener on `airpods`; every output the listener opened, judged."""
        port = free_port()
        hsd, jsd = FakeSoundDevice(), with_airpods(airpods)
        host = sa.HostSession(hsd, "Den", port=port, log=quiet, opus=False, stamps=True).start()
        join = sa.JoinSession(jsd, "127.0.0.1", "Saifan", output_query="AirPods", port=port, log=quiet).start()
        try:
            with gil_stress(stress):
                self.assertTrue(wait_until(lambda: join.info()["state"] == "in_sync", 10), join.info())
                time.sleep(seconds)
                info, hinfo = join.info(), host.info()
        finally:
            join.stop()
            host.stop()
        # each output from 3 s after its first sound: a reopened one's clock starts afresh, and averages out
        # the wobbling timestamps over a few seconds (its first sound is judged by the step tests)
        rs = [x for x in (heard(out, stamps_of(host.host), settle=3.0) for out in jsd.outputs) if x["good_s"] > 0.5]
        longest = max(rs, key=lambda x: x["good_s"])
        r = {"glitches": sum(x["glitches"] for x in rs), "err_p99_ms": max(x["err_p99_ms"] for x in rs),
             "err_median_ms": longest["err_median_ms"], "parts": rs}
        print(f"\n  listener on {airpods}: {r}, fake {[o.stats for o in jsd.outputs]}", file=sys.stderr)
        return info, hinfo, r, jsd, hsd.inputs[-1], join.devices

    def test_listener_on_airpods_macos_skips_stays_exact(self):
        """Saifan's Mac drops an IO cycle every few seconds by itself: each is one short gap, counted, and
        every sample after it is exactly on time (1.1.5-1.1.7 fell further behind with every one). After
        three in 30 s with SyncAudio's callback taking microseconds, it's macOS: the output is reopened
        with twice the buffer, and that's remembered for these AirPods. (When exactly is LadderTest's
        business: here a shared machine may stretch a callback past 1 ms and put it off.)"""
        info, hinfo, r, jsd, inp, devices = self.listen("Saifan’s AirPods Pro", 12)
        first, *later = jsd.outputs
        self.assertEqual(first.blocksize, 2048)
        levels = [e for e in info["events"] if e["event"] == "plan_level"]
        self.assertEqual([(e["from"], e["to"], e["reason"]) for e in levels], [(0, 1, "macos_skipping")] * len(later))
        self.assertEqual([o.blocksize for o in later], [4096] * len(levels))
        if levels:
            self.assertEqual(info["io"], [4096, 4096])
            self.assertEqual(devices["Saifan’s AirPods Pro"]["io_level"], 1)     # (no macOS helper: by name)
        forced = sum(o.stats["forced"] for o in jsd.outputs)
        late = sum(o.stats["late"] for o in jsd.outputs)
        self.assertGreaterEqual(info["skips"], forced - 1)                      # (the last may not be read yet)
        self.assertLessEqual(info["skips"], forced + late)
        # one gap per dropped cycle at most, and one for a reopen
        self.assertLessEqual(r["glitches"], missed(*jsd.outputs, inp) + len(later), r)
        self.assertLess(r["err_p99_ms"], 1.0, r)
        self.assertLess(abs(r["err_median_ms"]), 0.5, r)       # (and the clock sync's, on a busy process)
        steps = [e for e in hinfo["events"] if e["event"] == "delay_change"]
        self.assertEqual([(e["request"], e["to_ms"]) for e in steps], [("join", 590.0)])   # room made, once
        self.assertEqual(info["resyncs"], 0)

    def test_busy_mac_on_airpods_plays_exactly(self):
        """Python kept busy all along (gil_stress "app"): nothing of it reaches the sound."""
        info, hinfo, r, jsd, inp, _ = self.listen(AIRPODS["name"], 10, stress="app")
        out, = jsd.outputs
        self.assertLessEqual(r["glitches"], missed(out, inp), r)
        self.assertLess(r["err_p99_ms"], 1.0, r)
        self.assertLessEqual(info["skips"], out.stats["late"])                  # only the fake's own late wake-ups
        self.assertEqual((info["underruns"], info["resyncs"]), (0, 0), info)


class WatchdogActionTest(unittest.TestCase):
    """What a session does with what its output's watchdog says (when it says it: test_plan.LadderTest)."""

    def setUp(self):
        self.stack = hermetic()
        self.stack.__enter__()
        self.events = []
        self.join = sa.JoinSession(FakeSoundDevice(), "127.0.0.1", "Sam", log=quiet)
        self.join.output_name = AIRPODS["name"]
        self.join.system = types.SimpleNamespace(available=True, current=lambda: "MacBook Pro Speakers")
        self.join.client = types.SimpleNamespace(reopen_output=mock.Mock(),
                                                 event=lambda kind, **f: self.events.append((kind, f)))
        self.out = types.SimpleNamespace(name=AIRPODS["name"], facts=None)

    def tearDown(self):
        self.stack.close()

    def act(self, **action):
        self.join._act(action, self.out, self.join.client.event, self.join._reopen)
        return self.join.client.reopen_output

    def test_a_bigger_buffer_is_remembered_for_that_device(self):
        reopen = self.act(action="reopen", level=1, **{"from": 0}, reason="macos_skipping", persist=True)
        reopen.assert_called_once_with(level=1, reason="macos_skipping")
        self.assertEqual(self.join.devices[AIRPODS["name"]]["io_level"], 1)
        self.assertEqual(self.events, [("plan_level", {"from": 0, "to": 1, "reason": "macos_skipping"})])

    def test_a_reopen_for_this_time_only(self):
        self.act(action="reopen", level=0, **{"from": 0}, reason="rate_changed", persist=False)
        self.assertNotIn("io_level", self.join.devices.get(AIRPODS["name"], {}))

    def test_a_device_gone_follows_the_macs_output(self):
        reopen = self.act(action="follow", reason="device_gone")
        self.assertEqual(self.join.output_name, "MacBook Pro Speakers")
        (_, kw), = reopen.call_args_list
        self.assertEqual(kw["reason"], "device_gone")
        self.assertTrue(callable(kw["before_open"]))                    # PortAudio restarted first

    def test_a_warning_is_an_event(self):
        self.act(action="warn", warning="output_dropping", count=4)
        self.assertEqual(self.events, [("output_dropping", {"count": 4})])
        self.join.client.reopen_output.assert_not_called()

    def test_following_keeps_an_output_asked_for_and_never_takes_a_combined_one(self):
        self.join.output_query = self.join.output_name = "BlackHole 2ch"               # join --output (CI P8a)
        self.act(action="follow", reason="no_callbacks")
        self.assertEqual(self.join.output_name, "BlackHole 2ch")
        self.join.output_query, self.join.output_name = None, AIRPODS["name"]
        self.join.system.current = lambda: "Multi-Output Device"
        self.act(action="follow", reason="no_callbacks")
        self.assertEqual(self.join.output_name, AIRPODS["name"])     # (gone: pick_output's choice, never that)

    def test_an_output_that_wont_open_is_tried_again(self):
        self.join.client.reopen_output.side_effect = [OSError("still connecting"), None]
        self.act(action="follow", reason="device_gone")
        self.assertEqual(self.events, [("output_failed", {"error": "still connecting"})])
        self.join._reopen_client(reason="retry")
        self.assertFalse(self.join._failing)


class EventsTest(unittest.TestCase):
    """The events a session logs (spec 11.3) from what its output and clock report."""

    def setUp(self):
        self.events = []
        self.event = lambda kind, **f: self.events.append((kind, f))

    def test_each_second_notes_skips_running_dry_a_new_clock_line_and_a_new_latency(self):
        before = {"skips": 2, "skipped_ms": 85.3, "underruns": 0, "underrun_ms": 0.0, "dll_reanchors": 1,
                  "latency_changes": 0}
        now = {"skips": 4, "skipped_ms": 170.6, "underruns": 1, "underrun_ms": 40.0, "dll_reanchors": 2,
               "latency_changes": 1}
        sa.note_second(self.event, before, now)
        self.assertEqual(self.events, [("skip_burst", {"count": 2, "ms": 85.3}), ("underrun", {"ms": 40.0}),
                                       ("dll_reanchor", {}), ("latency_change", {"count": 1})])
        self.events.clear()
        sa.note_second(self.event, now, {"skips": 4, "dll_reanchors": 0, "latency_changes": 0})    # a reopened output
        self.assertEqual(self.events, [])

    def test_an_output_in_a_call_and_a_jump_of_the_clock(self):
        s = sa._Session()
        s.devices, s.settings = {}, sa.Settings()
        out = types.SimpleNamespace(name="AirPods", facts={"uid": "AirPods-uid"}, reason="start", call_profile=True,
                                    plan=types.SimpleNamespace(level=0))
        s._opened(out, self.event)
        self.assertEqual(self.events, [("output_open", {"name": "AirPods", "uid": "AirPods-uid", "reason": "start",
                                                        "level": 0}), ("call_profile", {"name": "AirPods"})])
        self.assertEqual(s.settings.device_key, "AirPods-uid")
        client = sa.Client("127.0.0.1", lambda *a, **k: None, log=quiet)
        client.second()
        client.clock.steps, client.clock.last_step = 1, -0.0254
        client.second()
        self.assertEqual([(e["event"], e.get("ms")) for e in client.events], [("clock_step", -25.4)])

    def test_the_sharing_mac_follows_its_own_choice_not_blackhole(self):
        # While sharing, the Mac's output is BlackHole: following it would play the share back into its
        # own capture (an echo for everyone). The output it had before (kept by _follow_output) instead.
        host = sa.HostSession(FakeSoundDevice(), "Den", log=quiet)
        host.system = types.SimpleNamespace(available=True, current=lambda: "BlackHole 2ch", previous=AIRPODS["name"])
        self.assertEqual(host._follow_target(), AIRPODS["name"])
        host.system.current = lambda: "MacBook Pro Speakers"                           # not sharing BlackHole
        self.assertEqual(host._follow_target(), "MacBook Pro Speakers")
        host.system.current, host.system.previous = (lambda: "Multi-Output Device"), None
        self.assertIsNone(host._follow_target())                                     # pick_output's choice
        host.output_query = "SyncTestBT 2ch"                                          # host --output (CI P8b)
        self.assertEqual(host._follow_target(), "SyncTestBT 2ch")


def gil_stress(kind):
    if not kind:
        return contextlib.nullcontext()
    import gil_stress as gs
    return gs.Stress(kind)


@unittest.skipUnless(has_fake, "needs the test build of libsacore")
class AirPodsTest(unittest.TestCase):
    """Putting AirPods in moves the Mac's output; SyncAudio should follow it."""

    @staticmethod
    def fake_switch_tool(state):
        """Stands in for SwitchAudioSource: reads and sets the Mac's output."""
        def run(_self, *args):
            if args[0] == "-c":
                return state["output"]
            state["output"] = args[1]
            return ""
        return mock.patch.object(sa.SystemOutput, "_run", run)

    def connect_airpods(self, sd, state):
        sd.DEVICES = sd.DEVICES + [AIRPODS]
        state["output"] = AIRPODS["name"]               # what macOS does when they connect

    def test_listener_moves_to_airpods(self):
        port = free_port()
        with hermetic():
            host = sa.HostSession(FakeSoundDevice(), "Den", port=port, log=quiet).start()
            sd, state = FakeSoundDevice(), {"output": "MacBook Pro Speakers"}
            with self.fake_switch_tool(state):
                join = sa.JoinSession(sd, "127.0.0.1", "Sam", port=port, switch_tool="SwitchAudioSource",
                                      log=quiet).start()
                try:
                    self.assertTrue(wait_until(lambda: join.info()["state"] == "in_sync"))
                    self.connect_airpods(sd, state)
                    self.assertTrue(wait_until(lambda: join.info()["output"] == AIRPODS["name"]
                                               and join.info()["state"] == "in_sync", 12), join.info())
                    airpods = sd.outputs[-1]
                    self.assertEqual(airpods.name, AIRPODS["name"])
                    self.assertFalse(sd.outputs[0].active, "the speakers were let go")
                    self.assertEqual(join.info()["audio"]["output"]["plan"]["blocksize"], 2048)
                    self.assertTrue(wait_until(lambda: np.abs(airpods.live(0.5)).max() > 0.1))
                finally:
                    join.stop()
                    host.stop()

    def test_sharing_mac_whose_airpods_vanish_plays_on_its_speakers(self):
        # AirPods back in their case: macOS's output stays BlackHole (sharing), so only the watchdog sees it
        sd, state = FakeSoundDevice(), {"output": "MacBook Pro Speakers"}
        logs = []
        with hermetic(), self.fake_switch_tool(state):
            host = sa.HostSession(sd, "Den", port=free_port(), switch_tool="SwitchAudioSource", log=logs.append).start()
            try:
                self.connect_airpods(sd, state)
                self.assertTrue(wait_until(lambda: getattr(host.output, "name", None) == AIRPODS["name"]))
                self.assertTrue(wait_until(lambda: state["output"] == "BlackHole 2ch"))
                airpods = sd.outputs[-1]
                airpods.abort()                         # no more callbacks, though PortAudio says it's active
                airpods.active = True
                sd.DEVICES = [d for d in sd.DEVICES if d["name"] != AIRPODS["name"]]
                self.assertTrue(wait_until(lambda: getattr(host.output, "name", None) == "MacBook Pro Speakers", 6),
                                (getattr(host.output, "name", None), logs[-5:]))
                self.assertEqual(host.source.name, "BlackHole 2ch")
            finally:
                host.stop()

    def test_after_the_airpods_vanished_stop_gives_the_sound_to_the_speakers(self):
        # Like the real SwitchAudioSource, the fake can only switch to devices that exist
        sd, state = FakeSoundDevice(), {"output": "MacBook Pro Speakers"}

        def run(_self, *args):
            if args[0] == "-c":
                return state["output"]
            if any(d["name"] == args[1] for d in sd.DEVICES):
                state["output"] = args[1]
            return ""
        with hermetic(), mock.patch.object(sa.SystemOutput, "_run", run):
            host = sa.HostSession(sd, "Den", port=free_port(), switch_tool="SwitchAudioSource", log=quiet).start()
            try:
                self.connect_airpods(sd, state)
                self.assertTrue(wait_until(lambda: getattr(host.output, "name", None) == AIRPODS["name"]))
                self.assertTrue(wait_until(lambda: state["output"] == "BlackHole 2ch"))
                airpods = sd.outputs[-1]
                airpods.abort()                         # back in their case
                airpods.active = True
                sd.DEVICES = [d for d in sd.DEVICES if d["name"] != AIRPODS["name"]]
                self.assertTrue(wait_until(lambda: getattr(host.output, "name", None) == "MacBook Pro Speakers", 6))
                self.assertEqual(host.system.previous, "MacBook Pro Speakers")
            finally:
                host.stop()
        self.assertEqual(state["output"], "MacBook Pro Speakers", "not BlackHole: no app could be heard")

    def test_an_output_that_fails_to_open_is_tried_again(self):
        class FlakySD(FakeSoundDevice):
            fail = 1                                    # the first open of the AirPods: still connecting

            def _StreamBase(self, kind, samplerate, blocksize, device, *a, **kw):
                if self.DEVICES[device]["name"] == AIRPODS["name"] and self.fail:
                    self.fail -= 1
                    raise self.PortAudioError("Internal PortAudio error [PaErrorCode -9986]")
                return super()._StreamBase(kind, samplerate, blocksize, device, *a, **kw)
        sd, state = FlakySD(), {"output": "MacBook Pro Speakers"}
        with hermetic(), self.fake_switch_tool(state):
            host = sa.HostSession(sd, "Den", port=free_port(), switch_tool="SwitchAudioSource", log=quiet).start()
            try:
                self.connect_airpods(sd, state)
                self.assertTrue(wait_until(lambda: any(e["event"] == "output_failed" for e in host.info()["events"])))
                self.assertTrue(wait_until(lambda: getattr(host.output, "name", None) == AIRPODS["name"], 8))
                self.assertTrue(wait_until(lambda: host.info()["local"]["state"] == "in_sync"))
            finally:
                host.stop()

    def test_listener_starts_on_the_macs_output_even_if_portaudio_is_behind(self):
        # PortAudio's default output is the one it saw when it started; macOS's helper knows better
        port = free_port()
        names = {40: "MacBook Pro Speakers", 41: AIRPODS["name"]}
        counter = types.SimpleNamespace(value=0, close=lambda: None)
        mac = types.SimpleNamespace(SEL={"default_output": 1, "global": 2}, system=1, default_output=lambda: 41,
                                    device_info_id=lambda i, out: {"id": i, "name": names[i], "uid": f"uid{i}"},
                                    device_info=lambda name, out: None, listen=lambda *a: counter)
        with hermetic():
            host = sa.HostSession(FakeSoundDevice(), "Den", port=port, log=quiet, opus=False).start()
            sd = with_airpods()
            sd.default.device = [0, 2]                  # the speakers, as PortAudio last saw it
            with mock.patch.object(sacore, "mac", mac):
                join = sa.JoinSession(sd, "127.0.0.1", "Saifan", port=port, log=quiet).start()
                try:
                    self.assertTrue(wait_until(lambda: join.info()["output"] == AIRPODS["name"]
                                               and join.info()["state"] == "in_sync", 12), join.info()["output"])
                finally:
                    join.stop()
                    host.stop()

    def test_sharing_mac_moves_to_airpods_and_keeps_sharing(self):
        sd, state = FakeSoundDevice(), {"output": "MacBook Pro Speakers"}
        with hermetic(), self.fake_switch_tool(state):
            host = sa.HostSession(sd, "Den", port=free_port(), switch_tool="SwitchAudioSource", log=quiet).start()
            try:
                self.assertEqual(state["output"], "BlackHole 2ch")     # apps now play into SyncAudio
                first_input = host.source
                self.connect_airpods(sd, state)
                self.assertTrue(wait_until(lambda: getattr(host.output, "name", None) == AIRPODS["name"]))
                self.assertTrue(wait_until(lambda: state["output"] == "BlackHole 2ch"),
                                "the Mac's output is put back on BlackHole so everyone keeps hearing it")
                self.assertEqual(host.system.previous, AIRPODS["name"])
                self.assertIsNot(host.source, first_input)
                self.assertTrue(wait_until(lambda: host.info()["local"]["state"] == "in_sync"))
                self.assertTrue(wait_until(lambda: np.abs(sd.outputs[-1].live(0.5)).max() > 0.1))
                gap = host.info()["capture"]["cap_gap_frames"]
                self.assertGreater(gap, 0)                              # capture carried on in real time
            finally:
                host.stop()
        self.assertEqual(state["output"], AIRPODS["name"], "stopping hands the sound back to the AirPods")


@unittest.skipUnless(has_fake, "needs the test build of libsacore")
class CombinedDeviceTest(unittest.TestCase):
    """People make a "Multi-Output Device" (often BlackHole + speakers) from BlackHole guides."""

    MULTI = {"name": "Multi-Output Device", "max_input_channels": 0, "max_output_channels": 2,
             "default_samplerate": 48000.0}

    def test_sharing_mac_puts_the_output_back_on_blackhole(self):
        sd, state = FakeSoundDevice(), {"output": "MacBook Pro Speakers"}
        sd.DEVICES = sd.DEVICES + [self.MULTI]
        with hermetic(), AirPodsTest.fake_switch_tool(state):
            host = sa.HostSession(sd, "Den", port=free_port(), switch_tool="SwitchAudioSource",
                                  log=lambda *a: None).start()
            try:
                state["output"] = "Multi-Output Device"         # picked in Control Center
                self.assertTrue(wait_until(lambda: state["output"] == "BlackHole 2ch", 6), state)
                self.assertEqual(host.output.name, "MacBook Pro Speakers", "never plays into it: echo")
                self.assertEqual(host.system.previous, "MacBook Pro Speakers")
            finally:
                host.stop()
        self.assertEqual(state["output"], "MacBook Pro Speakers")

    def test_never_picked_for_playback(self):
        sd = FakeSoundDevice()
        sd.DEVICES = sd.DEVICES + [self.MULTI]
        sd.default.device = [0, len(sd.DEVICES) - 1]
        self.assertEqual(sd.DEVICES[sa.pick_output(sd, None)]["name"], "MacBook Pro Speakers")


@unittest.skipUnless(has_fake, "needs the test build of libsacore")
class CaptureRecoveryTest(unittest.TestCase):
    def test_capture_comes_back_after_blackhole_does(self):
        # coreaudiod restarting: BlackHole stops calling back and is gone for 3 s. The first reopen fails;
        # it is tried again (with PortAudio restarted) until it works, everyone hearing silence meanwhile
        sd, logs = FakeSoundDevice(40), []
        with hermetic():
            host = sa.HostSession(sd, "Den", port=free_port(), log=logs.append, opus=False).start()
            try:
                time.sleep(2)
                blackhole = sd.inputs[-1]
                blackhole.abort()                       # no more callbacks, though PortAudio says active
                blackhole.active = True
                saved = sd.DEVICES
                sd.DEVICES = [d for d in saved if "BlackHole" not in d["name"]]
                failed = lambda: any(e["event"] == "capture_failed" for e in host.info()["events"])    # noqa: E731
                self.assertTrue(wait_until(failed, 4))
                time.sleep(2)
                sd.DEVICES = saved
                self.assertTrue(wait_until(lambda: sd.inputs[-1].active and host.source.stream is sd.inputs[-1], 8),
                                logs[-4:])
                self.assertTrue(wait_until(lambda: host.info()["local"]["state"] == "in_sync", 10))
                self.assertEqual(sum("couldn't capture" in m for m in logs), 1)         # said once
            finally:
                host.stop()


class StopTest(unittest.TestCase):
    """Stop while a session thread is half way through something: nothing may happen after it."""

    def test_a_listener_stopped_while_the_host_answers_opens_nothing(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen()

        def slow_host():                    # answers HELLO a second late
            conn, _ = srv.accept()
            sa.recv_frame(conn)
            time.sleep(1.0)
            try:
                conn.sendall(sa.frame(sa.HEADER, json.dumps({"rate": 48000, "channels": 2, "delay": 0.45, "name": "Den",
                                                             "version": "2.0.0", "proto": 2, "codec": "pcm"}).encode()))
            except OSError:
                pass
            time.sleep(1.5)
            conn.close()
        threading.Thread(target=slow_host, daemon=True).start()
        opened = []
        client = sa.Client("127.0.0.1", lambda *a, **kw: opened.append(a), port=srv.getsockname()[1], log=quiet)
        client.start()
        time.sleep(0.4)                     # HELLO sent, waiting for the answer
        client.stop()
        time.sleep(1.5)
        srv.close()
        self.assertEqual((opened, client.output), ([], None))
        client.reopen_output(before_open=lambda: opened.append("PortAudio restarted"))
        self.assertEqual(opened, [])

    def test_stop_while_following_an_output_never_leaves_the_mac_on_blackhole(self):
        # AirPods connect just before Stop: the tick thread is asking macOS for its output when Stop gives
        # the Mac its output back. It must not then point it at BlackHole again
        sd, state = with_airpods(), {"output": "MacBook Pro Speakers"}
        in_follow, stopped = threading.Event(), threading.Event()

        def run(_self, *args):
            if args[0] == "-c":
                if threading.current_thread().name == "host-tick" and state["output"] == AIRPODS["name"]:
                    in_follow.set()
                    stopped.wait(5)
                return state["output"]
            state["output"] = args[1]
            return ""
        with hermetic(), mock.patch.object(sa.SystemOutput, "_run", run):
            host = sa.HostSession(sd, "Den", port=free_port(), switch_tool="SwitchAudioSource", log=quiet).start()
            self.assertEqual(state["output"], "BlackHole 2ch")
            state["output"] = AIRPODS["name"]
            self.assertTrue(in_follow.wait(5))
            host.stop()
            stopped.set()
            time.sleep(1.5)
        self.assertNotEqual(state["output"], "BlackHole 2ch", "no app could be heard")


if __name__ == "__main__":
    unittest.main()
