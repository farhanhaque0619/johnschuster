"""Tests for syncaudio. No audio hardware needed: speakers are simulated.

    python3 -m unittest discover tests
"""
import contextlib
import heapq
import io
import os
import socket
import sys
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import syncaudio as sa  # noqa: E402

RATE = 48000


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class ClockSyncTest(unittest.TestCase):
    def test_tracks_offset_and_drift_through_lopsided_wifi(self):
        rng = np.random.default_rng(3)
        offset, drift = 1234.5, 80e-6           # client clock = offset + T * (1 + drift)
        local = lambda T: offset + T * (1 + drift)
        clock = sa.ClockSync()
        T, errors = 0.0, []
        while T < 60:
            up = 0.0015 + rng.exponential(0.003) + (0.08 if rng.random() < 0.05 else 0)
            down = 0.001 + rng.exponential(0.002)
            t1, t2 = local(T), T + up
            t3 = t2 + 0.0001
            t4 = local(t3 + down)
            clock.add(t1, t2, t3, t4)
            T = t3 + down + 0.2
            if T > 10:
                errors.append(clock.to_server(local(T + 0.1)) - (T + 0.1))
        errors = np.abs(errors)
        self.assertLess(np.percentile(errors, 95), 0.0008)
        self.assertLess(errors.max(), 0.0015)

    def test_unsynced_until_enough_pings(self):
        clock = sa.ClockSync()
        self.assertIsNone(clock.to_server(1.0))
        for i in range(sa.ClockSync.MIN_SAMPLES):
            clock.add(i, i + 0.001, i + 0.001, i + 0.002)
        self.assertAlmostEqual(clock.to_server(100.0), 100.0, places=6)


def simulate(seconds=45.0, delay=0.12, client_clock_ppm=80.0, host_dev_ppm=-35.0,
             client_dev_ppm=60.0, stall_at=None, seed=1):
    """Discrete-event simulation of one host and one friend.

    Everything runs on 'true' time T. The host clock is T itself, the friend's
    clock runs fast and is offset, both speakers run off their own crystals,
    and the network is jittery with lopsided delays. Returns, per speaker, the
    timing error of every output buffer (seconds; + = played early)."""
    rng = np.random.default_rng(seed)
    block = RATE * sa.BLOCK_MS // 1000
    local = lambda T: 777.0 + T * (1 + client_clock_ppm * 1e-6)
    clock = sa.ClockSync()
    timeline = sa.Timeline(RATE)
    players = {"host": sa.Player(RATE, 2, lambda t: t), "friend": sa.Player(RATE, 2, clock.to_server)}
    to_local = {"host": lambda T: T, "friend": local}
    devices = {"host": (512, host_dev_ppm, 0.025), "friend": (470, client_dev_ppm, 0.035)}
    errors = {"host": [], "friend": []}

    events, seq = [], 0

    def at(T, kind, data=None):
        nonlocal seq
        seq += 1
        heapq.heappush(events, (T, seq, kind, data))

    at(0.0, "capture", 0)
    at(0.0, "ping")
    for name, (_, _, latency) in devices.items():
        at(0.3, "callback", (name, 0, 0.3 + latency))

    last_arrival = 0.0
    while events:
        T, _, kind, data = heapq.heappop(events)
        if T > seconds:
            break
        if kind == "capture":                   # BlackHole is locked to the host clock
            k = data
            pcm = np.full((block, 2), 0.1, np.float32)
            n, captured = timeline.stamp(block, k * block / RATE)
            players["host"].feed(n, captured + delay, pcm)
            arrival = T + block / RATE + 0.002 + rng.exponential(0.006)
            if stall_at and stall_at <= T < stall_at + 0.3:
                arrival = stall_at + 0.3 + 0.005    # Wi-Fi hiccup: everything waits
            last_arrival = max(last_arrival, arrival)  # TCP keeps order
            at(last_arrival, "deliver", (n, captured + delay, pcm))
            at((k + 1) * block / RATE, "capture", k + 1)
        elif kind == "deliver":
            players["friend"].feed(*data)
        elif kind == "ping":
            up = 0.0015 + rng.exponential(0.003)
            down = 0.001 + rng.exponential(0.002)
            t1, t2 = local(T), T + up
            clock.add(t1, t2, t2 + 1e-4, local(t2 + 1e-4 + down))
            at(T + (0.03 if T < 2 else 0.2), "ping")
        elif kind == "callback":
            name, i, dac0 = data
            frames, ppm, latency = devices[name]
            dac = dac0 + i * frames / (RATE * (1 + ppm * 1e-6))
            p = players[name]
            before = p.read_n
            out = p.render(frames, to_local[name](dac))
            assert out.shape == (frames, 2)
            if before is not None and p.read_n is not None:
                intended = (dac - delay) * RATE     # sample index meant to be playing at dac
                errors[name].append((dac, (before - intended) / RATE))
            next_dac = dac0 + (i + 1) * frames / (RATE * (1 + ppm * 1e-6))
            at(next_dac - latency, "callback", (name, i + 1, dac0))
    return players, {k: np.array(v) for k, v in errors.items()}


class PlayerSimulationTest(unittest.TestCase):
    def assert_tight(self, errs, after, limit, label, skip=None):
        e = errs[errs[:, 0] > after]
        if skip:
            e = e[(e[:, 0] < skip[0]) | (e[:, 0] > skip[1])]
        worst = np.abs(e[:, 1]).max()
        self.assertLess(worst, limit, f"{label}: worst error {worst * 1000:.2f} ms")
        return worst

    def test_two_macs_stay_in_sync_despite_drift(self):
        players, errs = simulate()
        h = self.assert_tight(errs["host"], 10, 0.0005, "host")
        f = self.assert_tight(errs["friend"], 10, 0.0015, "friend")
        print(f"\n  simulated worst error after 10 s: host {h * 1000:.2f} ms, friend {f * 1000:.2f} ms")
        self.assertEqual(players["host"].resyncs, 0)
        self.assertEqual(players["friend"].resyncs, 0)
        self.assertEqual(players["host"].dropouts, 0)
        self.assertEqual(players["friend"].dropouts, 0)

    def test_extreme_drift(self):
        players, errs = simulate(client_clock_ppm=-400, host_dev_ppm=300, client_dev_ppm=-300, seed=7)
        self.assert_tight(errs["host"], 15, 0.001, "host")
        self.assert_tight(errs["friend"], 15, 0.002, "friend")

    def test_recovers_from_wifi_stall(self):
        players, errs = simulate(stall_at=20.0, seed=2)
        self.assertGreaterEqual(players["friend"].dropouts, 1)
        # Timing never slips: the gap is played as silence, not by falling behind.
        self.assert_tight(errs["friend"], 10, 0.0015, "friend")
        self.assertEqual(players["friend"].resyncs, 0)


class PlayerTest(unittest.TestCase):
    def test_plays_each_sample_at_its_time(self):
        p = sa.Player(RATE, 2, lambda t: t)
        ramp = np.arange(RATE, dtype=np.float32)[:, None].repeat(2, 1) / RATE
        p.feed(0, 10.0, ramp)                   # sample 0 is due at t=10
        silent = p.render(100, 9.0)
        self.assertFalse(silent.any())          # too early: silence
        out = p.render(100, 10.5)               # first frame due at 10.5 -> sample 24000
        self.assertAlmostEqual(out[0, 0], 24000 / RATE, places=4)

    def test_channel_modes_and_volume(self):
        settings = sa.Settings()
        p = sa.Player(RATE, 2, lambda t: t, settings)
        pcm = np.zeros((RATE, 2), np.float32)
        pcm[:, 0], pcm[:, 1] = 0.2, 0.6
        p.feed(0, 0.0, pcm)
        np.testing.assert_allclose(p.render(10, 0.1)[0], [0.2, 0.6])
        settings.channel = "right"
        np.testing.assert_allclose(p.render(10, 0.2)[0], [0.6, 0.6])
        settings.channel, settings.volume = "left", 0.5
        np.testing.assert_allclose(p.render(10, 0.3)[0], [0.1, 0.1])
        self.assertEqual(p.render(10, 0.4, out_channels=1).shape, (10, 1))

    def test_offset_plays_later(self):
        settings = sa.Settings(offset_ms=10)
        p = sa.Player(RATE, 2, lambda t: t, settings)
        p.feed(0, 0.0, np.arange(RATE, dtype=np.float32)[:, None].repeat(2, 1))
        self.assertEqual(p.render(10, 0.5)[0, 0], 0.49 * RATE)

    def test_new_stream_resets(self):
        p = sa.Player(RATE, 2, lambda t: t)
        p.feed(1000, 5.0, np.ones((240, 2), np.float32))
        p.feed(0, 8.0, np.ones((240, 2), np.float32))   # host restarted
        self.assertEqual((p.n_first, p.n_end), (0, 240))


class TimelineTest(unittest.TestCase):
    def test_smooths_jitter_and_follows_jumps(self):
        rng = np.random.default_rng(0)
        tl = sa.Timeline(RATE)
        block = 240
        for k in range(2000):
            n, t = tl.stamp(block, 50.0 + k * block / RATE + rng.normal(0, 0.001))
            if k > 500:     # settled: 1 ms of callback jitter is smoothed away
                self.assertLess(abs(t - (50.0 + n / RATE)), 0.0003)
        for k in range(2000, 2010):                 # the device stalls for 100 ms
            n, t = tl.stamp(block, 50.1 + k * block / RATE)
        self.assertLess(abs(t - (50.1 + n / RATE)), 0.002)

    def test_follows_a_capture_device_with_its_own_clock(self):
        tl = sa.Timeline(RATE)
        block, fast = 240, 1 + 200e-6               # e.g. a USB interface running 200 ppm fast
        for k in range(20000):                      # 100 s
            n, t = tl.stamp(block, 7.0 + k * block / (RATE * fast))
            if k > 2000:
                self.assertLess(abs(t - (7.0 + n / (RATE * fast))), 0.0005)


class FakeSpeaker:
    """Behaves like a sound card: contiguous buffers on its own timeline."""

    def __init__(self, rate, render, to_local, frames=480, latency=0.03):
        self.latency, self.frames, self.rate = latency, frames, rate
        self.render, self.to_local = render, to_local
        self.records = []
        self._stop = threading.Event()
        self.dac0 = time.monotonic() + latency + 0.02
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        k = 0
        while not self._stop.is_set():
            dac = self.dac0 + k * self.frames / self.rate
            wait = dac - self.latency - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            out = self.render(self.frames, self.to_local(dac), 2)
            self.records.append((dac, out[:, 0].copy()))
            k += 1

    def close(self):
        self._stop.set()

    def onsets(self):
        times = []
        quiet = 0
        for dac, block in list(self.records):
            loud = np.flatnonzero(np.abs(block) > 0.05)
            pos = 0
            for i in loud:
                if quiet + (i - pos) > 0.1 * self.rate:
                    times.append(dac + i / self.rate)
                quiet, pos = 0, i + 1
            quiet += len(block) - pos
        return np.array(times)


class EndToEndTest(unittest.TestCase):
    def test_host_and_friend_click_together(self):
        port, delay = free_port(), 0.12
        quiet = lambda *a: None
        host = sa.Host(RATE, 2, delay, port, "test-host", log=quiet)
        host.start()
        host.local = sa.Player(RATE, 2, lambda t: t)
        host_speaker = FakeSpeaker(RATE, host.local.render, lambda t: t, frames=512)

        friend_clock = lambda t: 5000.0 + t * (1 + 150e-6)   # another Mac's clock
        speakers = []

        def open_output(rate, render):
            speakers.append(FakeSpeaker(rate, render, friend_clock, frames=441, latency=0.04))
            return speakers[-1]

        client = sa.Client("127.0.0.1", open_output, port, "test-friend", log=quiet,
                           now=lambda: friend_clock(time.monotonic()))
        clicks = sa.ClickSource(RATE, 2, host.on_block)
        clicks.start()
        client.start()
        try:
            time.sleep(10)
            listeners = host.listeners()
        finally:
            clicks.close()
            client.stop()
            host_speaker.close()
            host.stop()

        self.assertEqual([(l["name"], l["state"]) for l in listeners], [("test-friend", "in_sync")])
        self.assertEqual(client.state, "playing")
        h, f = host_speaker.onsets(), speakers[0].onsets()
        cutoff = host.timeline.t0 + 5        # give clock sync a few seconds
        h, f = h[h > cutoff], f[f > cutoff]
        self.assertGreater(len(h), 8)
        self.assertGreater(len(f), 8)
        # Each click comes out of the host speaker when it was meant to...
        intended = host.timeline.t0 + delay + np.round((h - host.timeline.t0 - delay) * 2) / 2
        self.assertLess(np.abs(h - intended).max(), 0.001)
        # ...and out of the friend's speaker at the same moment.
        diffs = np.array([f[np.argmin(np.abs(f - t))] - t for t in h])
        print(f"\n  end-to-end host vs friend: median {np.median(np.abs(diffs)) * 1000:.2f} ms, "
              f"worst {np.abs(diffs).max() * 1000:.2f} ms")
        self.assertLess(np.abs(diffs).max(), 0.002)

    def test_discovery(self):
        port = free_port()
        host = sa.Host(RATE, 2, 0.1, port, "Living Room Mac", log=lambda *a: None)
        host.start()
        try:
            self.assertEqual(sa.discover(port, address="127.0.0.1"), ("127.0.0.1", "Living Room Mac"))
        finally:
            host.stop()


class FakeSoundDevice:
    """Just enough of the sounddevice module: BlackHole, a mic and speakers,
    with streams that call back on their own threads like CoreAudio does."""

    class PortAudioError(Exception):
        pass

    DEVICES = [
        {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2, "default_samplerate": 48000.0},
        {"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48000.0},
        {"name": "MacBook Pro Speakers", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000.0},
    ]

    def __init__(self):
        self.default = types.SimpleNamespace(device=[0, 0])     # output already on BlackHole
        self.outputs = []
        fake = self

        class Stream:
            latency = 0.03

            def __init__(self, samplerate, device, channels, dtype, callback, blocksize=0, latency=None):
                self.rate, self.channels, self.callback, self.device = samplerate, channels, callback, device
                self.frames = blocksize or 512
                self.stop_event = threading.Event()
                self.written = []

            def start(self):
                threading.Thread(target=self.run, daemon=True).start()

            def stop(self):
                self.stop_event.set()

            def close(self):
                pass

        class OutputStream(Stream):
            def __init__(self, **kw):
                super().__init__(**kw)
                fake.outputs.append(self)

            def run(self):
                epoch = 1000.0          # PortAudio's clock has its own epoch
                dac0, k = time.monotonic() + 0.05, 0
                while not self.stop_event.is_set():
                    dac = dac0 + k * self.frames / self.rate
                    time.sleep(max(0.0, dac - self.latency - time.monotonic()))
                    out = np.zeros((self.frames, self.channels), np.float32)
                    t = types.SimpleNamespace(currentTime=time.monotonic() + epoch, outputBufferDacTime=dac + epoch)
                    self.callback(out, self.frames, t, None)
                    self.written.append(out)
                    k += 1

        class InputStream(Stream):
            def run(self):
                t0, k = time.monotonic(), 0
                while not self.stop_event.is_set():
                    end = t0 + (k + 1) * self.frames / self.rate
                    time.sleep(max(0.0, end - time.monotonic()))
                    n = np.arange(k * self.frames, (k + 1) * self.frames)
                    pcm = np.sin(2 * np.pi * 440 * n / self.rate)[:, None].repeat(self.channels, 1) * 0.5
                    now = time.monotonic() + 1000.0
                    t = types.SimpleNamespace(currentTime=now, inputBufferAdcTime=now - (time.monotonic() - end) - self.frames / self.rate)
                    self.callback(pcm.astype(np.float32), self.frames, t, None)
                    k += 1

        self.OutputStream, self.InputStream = OutputStream, InputStream

    def query_devices(self, device=None, kind=None):
        return self.DEVICES if device is None else self.DEVICES[device]


class CommandLineTest(unittest.TestCase):
    def test_host_and_join_commands(self):
        hosting, joining = FakeSoundDevice(), FakeSoundDevice()
        port = free_port()
        quit_after = lambda seconds: lambda self: (threading.Timer(seconds, self.quit.set).start(), self)[1]
        out = io.StringIO()
        host = threading.Thread(target=sa.main, args=(["host", "--port", str(port), "--name", "Den"],))
        # Each "Mac" gets its own fake devices, however late its thread gets going.
        devices = lambda: hosting if threading.current_thread() is host else joining
        with mock.patch.object(sa.Console, "__enter__", quit_after(8)), \
                mock.patch.object(sa.shutil, "which", return_value=None), \
                mock.patch.object(sa, "import_sounddevice", side_effect=devices), \
                contextlib.redirect_stdout(out):
            host.start()
            time.sleep(1)
            sa.main(["join", "127.0.0.1", "--port", str(port), "--name", "Sam"])
            host.join(10)
        log = out.getvalue()
        self.assertIn('Set this Mac\'s sound output to "BlackHole 2ch"', log)
        self.assertIn("playing on: MacBook Pro Speakers", log)    # not BlackHole: no feedback loop
        self.assertIn("+ Sam joined", log)
        self.assertIn("connected to Den", log)
        for sd in (hosting, joining):
            played = np.concatenate(sd.outputs[0].written)
            self.assertGreater(np.abs(played).max(), 0.4, "the 440 Hz tone reached the speaker")


class AutoDelayTest(unittest.TestCase):
    def test_raises_at_once_when_a_listener_runs_short(self):
        auto = sa.AutoDelay()
        self.assertIsNone(auto.update(10, {"a": {"headroom_ms": 60, "dropouts": 0}}))
        new = auto.update(11, {"a": {"headroom_ms": -219, "dropouts": 3}, "b": {"headroom_ms": 80, "dropouts": 0}})
        self.assertAlmostEqual(new, 0.40)      # 150 ms + 219 ms late + 30 ms margin, rounded up
        # Listeners need a few seconds to measure the new delay; old bad numbers are ignored.
        self.assertIsNone(auto.update(13, {"a": {"headroom_ms": -219, "dropouts": 5}}))

    def test_dropouts_alone_raise_the_delay(self):
        auto = sa.AutoDelay()
        auto.update(0, {"a": {"headroom_ms": 40, "dropouts": 0}})
        self.assertAlmostEqual(auto.update(1, {"a": {"headroom_ms": 40, "dropouts": 1}}), 0.20)

    def test_lowers_slowly_after_a_long_calm(self):
        auto = sa.AutoDelay(0.4)
        changes = [(t, d) for t in range(1500)
                   if (d := auto.update(t, {"a": {"headroom_ms": 400, "dropouts": 0}})) is not None]
        self.assertGreaterEqual(changes[0][0], sa.AutoDelay.CALM)
        steps = np.diff([0.4] + [d for _, d in changes])
        self.assertTrue(np.all(steps < 0) and np.all(steps >= -0.0501), steps)
        self.assertGreaterEqual(changes[-1][1], sa.AutoDelay.MIN)

    def test_never_lowers_soon_after_trouble(self):
        auto = sa.AutoDelay()
        auto.update(0, {"a": {"headroom_ms": 5, "dropouts": 0}})        # raised
        lowered = [t for t in range(1, 600) if auto.update(t, {"a": {"headroom_ms": 400, "dropouts": 0}})]
        self.assertEqual(lowered, [])

    def test_waits_for_measurements(self):
        self.assertIsNone(sa.AutoDelay().update(10, {"a": {"headroom_ms": None, "dropouts": 0}}))
        self.assertIsNone(sa.AutoDelay().update(10, {}))


def wait_until(check, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.2)
    return False


class AirPodsTest(unittest.TestCase):
    """Putting AirPods in moves the Mac's output; SyncAudio should follow it."""

    AIRPODS = {"name": "Farhan’s AirPods Pro", "max_input_channels": 1, "max_output_channels": 2,
               "default_samplerate": 48000.0}
    quiet = staticmethod(lambda *a: None)

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
        sd.DEVICES = sd.DEVICES + [self.AIRPODS]
        state["output"] = self.AIRPODS["name"]          # what macOS does when they connect

    def test_listener_moves_to_airpods(self):
        port = free_port()
        with mock.patch.object(sa.shutil, "which", return_value=None):
            host = sa.HostSession(FakeSoundDevice(), "Den", port=port, log=self.quiet).start()
        sd, state = FakeSoundDevice(), {"output": "MacBook Pro Speakers"}
        with self.fake_switch_tool(state):
            join = sa.JoinSession(sd, "127.0.0.1", "Sam", port=port, switch_tool="SwitchAudioSource",
                                  log=self.quiet).start()
            try:
                self.assertTrue(wait_until(lambda: join.info()["state"] == "in_sync"))
                self.connect_airpods(sd, state)
                self.assertTrue(wait_until(lambda: join.info()["output"] == self.AIRPODS["name"]
                                           and join.info()["state"] == "in_sync"), join.info())
                airpods = sd.outputs[-1]
                self.assertEqual(airpods.device, len(sd.DEVICES) - 1)
                self.assertTrue(sd.outputs[0].stop_event.is_set(), "the speakers were let go")
                self.assertTrue(wait_until(lambda: len(airpods.written) > 20))
                self.assertGreater(np.abs(np.concatenate(airpods.written)).max(), 0.4)
            finally:
                join.stop()
                host.stop()

    def test_sharing_mac_moves_to_airpods_and_keeps_sharing(self):
        sd, state = FakeSoundDevice(), {"output": "MacBook Pro Speakers"}
        with self.fake_switch_tool(state):
            host = sa.HostSession(sd, "Den", port=free_port(), switch_tool="SwitchAudioSource",
                                  log=self.quiet).start()
            try:
                self.assertEqual(state["output"], "BlackHole 2ch")     # apps now play into SyncAudio
                first_input = host.source
                self.connect_airpods(sd, state)
                self.assertTrue(wait_until(lambda: host.output.name == self.AIRPODS["name"]))
                self.assertTrue(wait_until(lambda: state["output"] == "BlackHole 2ch"),
                                "the Mac's output is put back on BlackHole so everyone keeps hearing it")
                self.assertEqual(host.system.previous, self.AIRPODS["name"])
                self.assertIsNot(host.source, first_input)
                self.assertTrue(wait_until(lambda: host.info()["local"]["state"] == "in_sync"))
                self.assertTrue(wait_until(lambda: len(sd.outputs[-1].written) > 20))
                self.assertGreater(np.abs(np.concatenate(sd.outputs[-1].written)).max(), 0.4)
            finally:
                host.stop()
        self.assertEqual(state["output"], self.AIRPODS["name"], "stopping hands the sound back to the AirPods")


class GroupVolumeTest(unittest.TestCase):
    def test_changes_glide_over_one_block(self):
        host = sa.Host(RATE, 1, 0.1, free_port(), log=lambda *a: None)
        host.local = sa.Player(RATE, 1, lambda t: t)
        ones = np.ones((240, 1), np.float32)
        host.on_block(ones, 0.0)
        host.group_volume = 0.5
        host.on_block(ones, 240 / RATE)
        host.on_block(ones, 480 / RATE)
        played = host.local.ring[:720, 0]
        np.testing.assert_allclose(played[:240], 1.0)
        self.assertTrue(np.all(np.diff(played[240:480]) <= 0) and played[240] == 1.0 and played[479] == 0.5)
        np.testing.assert_allclose(played[480:], 0.5)


class SessionTest(unittest.TestCase):
    """HostSession + JoinSession, the pieces the app is built from."""

    def test_three_macs_and_a_live_delay_change(self):
        quiet = lambda *a: None
        port = free_port()
        host_sd = FakeSoundDevice()
        with mock.patch.object(sa.shutil, "which", return_value=None):
            host = sa.HostSession(host_sd, "Den", port=port, log=quiet).start()
        listeners = [sa.JoinSession(FakeSoundDevice(), "127.0.0.1", name, port=port, log=quiet).start()
                     for name in ("Sam", "Alex")]
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                info = host.info()
                if [l["state"] for l in info["listeners"]] == ["in_sync", "in_sync"]:
                    break
                time.sleep(0.25)
            self.assertEqual([(l["name"], l["state"]) for l in info["listeners"]],
                             [("Alex", "in_sync"), ("Sam", "in_sync")])
            self.assertTrue(info["auto_delay"])
            self.assertEqual(info["local"]["state"], "in_sync")
            self.assertEqual(info["capture"], "BlackHole 2ch")
            self.assertEqual(info["output"], "MacBook Pro Speakers")

            host.set_delay(300)
            time.sleep(4)
            for l in listeners:
                state = l.info()
                self.assertEqual(state["delay_ms"], 300)
                self.assertEqual(state["connection"], "connected")
                self.assertEqual(state["state"], "in_sync")
                self.assertGreater(state["headroom_ms"], 150)
            self.assertEqual(host.info()["delay_ms"], 300)
            self.assertFalse(host.info()["auto_delay"])

            # Sam turns everyone down to half.
            sam, alex = listeners
            sam.client.set_group_volume(0.5)
            time.sleep(2)
            self.assertEqual((host.info()["group_volume"], host.info()["group_volume_by"]), (0.5, "Sam"))
            self.assertEqual((alex.info()["group_volume"], alex.info()["group_volume_by"]), (0.5, "Sam"))
            for sd in [host_sd] + [l.sd for l in listeners]:
                recent = np.abs(np.concatenate(sd.outputs[-1].written[-40:])).max()
                self.assertAlmostEqual(recent, 0.25, delta=0.03)    # the 0.5 tone at half volume
        finally:
            for l in listeners:
                l.stop()
            host.stop()
        for sd in [host_sd] + [l.sd for l in listeners]:
            played = np.concatenate(sd.outputs[-1].written)
            self.assertGreater(np.abs(played).max(), 0.4, "the 440 Hz tone reached every speaker")

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


if __name__ == "__main__":
    unittest.main()
