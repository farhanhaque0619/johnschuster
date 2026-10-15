"""How 2.0 opens audio devices: the IO plan, what PortAudio makes of it, the streams it opens (only C
callbacks), the order it opens and closes them in, the self-heal ladder, and the self-test."""
import os
import sys
import threading
import time
import types
import unittest
from unittest import mock

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
sys.path.insert(0, here)
import sacore  # noqa: E402
import audio_engine as ae  # noqa: E402
import vdevice as vd  # noqa: E402

RATE = 48000


def pa_host_frames(latency, rate, fixed, blocksize, rng):
    """PortAudio v19.7.0's host buffer size on CoreAudio (pa_mac_core.c:1635-1697): the requested
    latency less the device's fixed latency, at least 1 frame, rounded up to a multiple of the user
    blocksize, clipped to the device's BufferFrameSizeRange."""
    res = max(int(latency * rate) - fixed, 0)          # UInt32 truncation, then "prevent negative"
    res = max(res, 1)
    if blocksize:
        res = -(-res // blocksize) * blocksize
    return min(max(res, rng[0]), rng[1])


AIRPODS = {"fixed": 7680, "range": (15, 4096)}          # ~160 ms fixed (device + safety + stream)
SPEAKERS = {"fixed": 1344, "range": (15, 4096)}         # ~28 ms


def device(name, fixed_frames, out=2, inp=0, **sim):
    """A device as PortAudio reports it: default_low = (fixed + max(64, smallest buffer)) / rate."""
    return {"name": name, "max_output_channels": out, "max_input_channels": inp,
            "default_low_output_latency": (fixed_frames + 64) / RATE,
            "default_high_output_latency": (fixed_frames + 512) / RATE,
            "default_low_input_latency": (fixed_frames + 64) / RATE, **sim}


def facts(fixed=0, transport="bltn", **kw):
    return {"latency": 0, "safety": fixed // 2, "stream_latency": fixed - fixed // 2, "nominal_rate": RATE,
            "actual_rate": RATE, "transport": transport, "alive": 1, "uid": "uid", "name": "x",
            "out_channels": 2, "buffer_frames": 0, "buffer_min": 15, "buffer_max": 4096, **kw}


class PortAudioModelTest(unittest.TestCase):
    def test_1_1_6_gave_airpods_the_smallest_buffer(self):
        # latency=0.1 minus ~160 ms of fixed latency is nothing: the device minimum (the static)
        self.assertEqual(pa_host_frames(0.1, RATE, AIRPODS["fixed"], 0, AIRPODS["range"]), 15)
        self.assertGreater(pa_host_frames(0.1, RATE, SPEAKERS["fixed"], 0, SPEAKERS["range"]), 3000)

    def test_an_explicit_blocksize_with_latency_0_is_the_host_buffer(self):
        for dev in (AIRPODS, SPEAKERS):
            for bs in (1024, 2048, 4096):
                self.assertEqual(pa_host_frames(ae.PA_LATENCY, RATE, dev["fixed"], bs, dev["range"]), bs)

    def test_every_plan_gets_its_blocksize_from_portaudio(self):
        sd = vd.VirtualSD(vd.Sim(), [device("AirPods", AIRPODS["fixed"]), device("Speakers", SPEAKERS["fixed"])])
        for i, dev, want in ((0, AIRPODS, 2048), (1, SPEAKERS, 1024)):
            plan = ae.plan_output(sd, i, RATE, None, 0)
            self.assertEqual(plan.blocksize, want)
            self.assertEqual(pa_host_frames(plan.latency, RATE, dev["fixed"], plan.blocksize, dev["range"]), want)
            up = ae.plan_output(sd, i, RATE, None, 1)
            self.assertEqual(pa_host_frames(up.latency, RATE, dev["fixed"], up.blocksize, dev["range"]),
                             min(2 * want, ae.OUT_BLOCK_MAX))
            auto = ae.plan_output(sd, i, RATE, None, 2)         # PortAudio's own size, but not tiny
            self.assertEqual(auto.blocksize, 0)
            self.assertGreaterEqual(pa_host_frames(auto.latency, RATE, dev["fixed"], 0, dev["range"]), 2000)


class PlanTest(unittest.TestCase):
    sd = vd.VirtualSD(vd.Sim(), [device("AirPods", AIRPODS["fixed"]), device("Speakers", SPEAKERS["fixed"])])

    def test_bluetooth_by_transport_or_by_latency(self):
        p = ae.plan_output(self.sd, 1, RATE, facts(0, "blue"), 0)
        self.assertEqual((p.blocksize, p.bt_class, p.source), (2048, True, "coreaudio"))
        p = ae.plan_output(self.sd, 1, RATE, facts(3840 * 2, "virt"), 0)      # the CI twin: by latency
        self.assertEqual((p.blocksize, p.bt_class), (2048, True))
        self.assertAlmostEqual(p.fixed_s, 0.16)
        p = ae.plan_output(self.sd, 1, RATE, facts(1344), 0)
        self.assertEqual((p.blocksize, p.latency, p.bt_class), (1024, 0.0, False))

    def test_without_the_mac_helper_portaudio_tells_the_fixed_latency(self):
        p = ae.plan_output(self.sd, 0, RATE, None, 0)
        self.assertEqual(p.source, "portaudio")
        self.assertAlmostEqual(p.fixed_s, 0.16, places=6)
        broken = types.SimpleNamespace(query_devices=mock.Mock(side_effect=OSError))
        p = ae.plan_output(broken, 0, RATE, None, 0)
        self.assertEqual((p.source, p.fixed_s, p.blocksize), ("none", None, 1024))
        self.assertEqual(ae.plan_output(broken, 0, RATE, None, 2).latency, 0.25)

    def test_levels(self):
        self.assertEqual(ae.plan_output(self.sd, 0, RATE, None, 1).blocksize, 4096)
        self.assertEqual(ae.plan_output(self.sd, 1, RATE, None, 1).blocksize, 2048)
        p = ae.plan_output(self.sd, 0, RATE, None, 2)
        self.assertEqual((p.blocksize, p.level), (0, 2))
        self.assertAlmostEqual(p.latency, 0.21, places=6)


class Player(ae.LocalPlayer):
    attached = None

    def attach(self, engine):
        super().attach(engine)
        self.attached = engine


SETTINGS = types.SimpleNamespace(offset_ms=0.0, volume=1.0, channel="both")


class OpenTest(unittest.TestCase):
    """AudioOutput / AudioInput on virtual devices (no threads: the test drives the clock)."""

    def setUp(self):
        self.sim = vd.Sim()
        self.sd = vd.VirtualSD(self.sim, [device("AirPods", 7680, lead=0.2), device("Speakers", 1344, lead=0.03),
                                          device("BlackHole 2ch", 0, inp=2, lead=0.005),
                                          device("Stingy", 0, lead=0.01, force_frames=256),
                                          device("Mono", 1344, out=1, lead=0.03)])
        self.outputs = []

    def tearDown(self):
        for o in self.outputs:
            o.close()

    def output(self, index, **kw):
        player = Player(RATE, 2)
        out = ae.AudioOutput(self.sd, index, RATE, player, SETTINGS, {}, log=lambda m: None,
                             now=lambda: self.sim.t, threads=False, **kw)
        self.outputs.append(out)
        self.sim.every(ae.OutputEngine.TICK, out.engine.tick, skip=lambda t: not out.stream.active)
        return out, player

    def test_i1_only_c_callbacks_with_an_explicit_buffer(self):
        self.output(0)
        self.output(1)
        ae.AudioInput(self.sd, 2, RATE, 2, lambda *a: None, now=lambda: self.sim.t, threads=False).close()
        self.assertEqual(len(self.sd.opened), 3)
        for s in self.sd.opened:
            self.assertIsInstance(s["callback"], int)
            self.assertIsNone(s["wrap_callback"])
            self.assertGreater(s["blocksize"], 0)
            self.assertEqual(s["latency"], 0.0)
            self.assertEqual(s["dtype"], "float32")
        self.assertEqual([s["blocksize"] for s in self.sd.opened], [2048, 1024, 512])
        self.assertEqual(self.sd.opened[0]["callback"], ctypes_address("sa_out_callback"))
        self.assertEqual(self.sd.opened[2]["callback"], ctypes_address("sa_in_callback"))

    def test_info_never_asks_portaudio(self):
        # The app polls info() from its own thread while a session may be restarting PortAudio, which
        # frees what such a question would read: PortAudio's figures are taken once, at open
        out, _ = self.output(0)
        inp = ae.AudioInput(self.sd, 2, RATE, 2, lambda *a: None, now=lambda: self.sim.t, threads=False)
        self.sim.advance(1.0)
        before = out.info()["pa"], inp.info()["pa"]
        self.sd.query_devices = self.sd.get_portaudio_version = mock.Mock(side_effect=AssertionError("PortAudio"))
        self.assertEqual((out.info()["pa"], inp.info()["pa"]), before)
        self.assertIn("stream_latency_ms", before[0])
        inp.close()

    def test_i3_the_ring_is_freed_last(self):
        events = []
        out, player = self.output(1)
        self.assertIs(player.attached, out.engine)
        stream, ring, engine = out.stream, out.ring, out.engine
        stream.abort = wrap(stream.abort, events, "abort")
        stream.close = wrap(stream.close, events, "close")
        engine.stop = wrap(engine.stop, events, "engine stopped")
        ring.close = wrap(ring.close, events, "ring freed")
        player.detach = wrap(player.detach, events, "detached")
        self.outputs.remove(out)
        out.close()
        self.assertEqual(events, ["detached", "engine stopped", "abort", "close", "ring freed"])

    def test_a_stream_that_wont_close_keeps_its_ring(self):
        # PortAudio couldn't stop the stream (sounddevice says so only if asked): it may still call back
        out, _ = self.output(1)
        out.stream.stuck = True
        self.outputs.remove(out)
        self.assertFalse(out.close())
        self.assertIsNotNone(out.ring.h)
        out.stream.stuck = False
        out.stream.abort()
        out.ring.close()

    def test_an_engine_that_wont_stop_keeps_its_ring_and_content(self):
        # a starved tick outlasting close(): what it is about to read must not be freed under it
        out, player = self.output(1)
        release = threading.Event()
        out.engine.tick = lambda: release.wait(5)
        out.engine.start()
        time.sleep(0.05)
        self.outputs.remove(out)
        with mock.patch.object(ae, "STOP_WAIT", 0.1):
            self.assertFalse(out.close())
            self.assertFalse(player.sb.close(lambda m: None))
        self.assertTrue(out.stream.closed)                              # the device is let go of anyway
        self.assertIsNotNone(out.ring.h)
        self.assertIsNotNone(player.sb.content.h)
        release.set()
        out.engine.thread.join(2)
        self.assertTrue(player.sb.close())                              # stopped now: it can go
        out.ring.close()

    def test_a_stream_that_fails_to_start_leaves_nothing_behind(self):
        # AirPods gone between open and start: the stream, its listeners and its ring are all let go
        listening, freed = [], []

        class Counter:
            value = 0

            def close(self):
                listening.remove(self)

        def listen(*_):
            listening.append(Counter())
            return listening[-1]
        mac = types.SimpleNamespace(SEL=dict.fromkeys(("overload", "latency", "nominal_rate", "alive", "global",
                                                       "wildcard", "output"), 1),
                                    stream_device=lambda *_: 42, device_info=lambda *_: None,
                                    device_info_id=lambda *_: None, listen=listen)
        ring_close = sacore.Ring.close

        def close(ring):
            freed.append(ring)
            ring_close(ring)
        with mock.patch.object(sacore, "mac", mac), mock.patch.object(sacore.Ring, "close", close), \
                mock.patch.object(vd.VirtualStream, "start", side_effect=vd.PortAudioError("Device unavailable")):
            with self.assertRaises(vd.PortAudioError):
                ae.AudioOutput(self.sd, 0, RATE, Player(RATE, 2), SETTINGS, {}, log=lambda m: None,
                               now=lambda: self.sim.t, threads=False)
            self.assertEqual((listening, len(freed), self.sd.streams[-1].closed), ([], 1, True))
            with self.assertRaises(vd.PortAudioError):
                ae.AudioInput(self.sd, 2, RATE, 2, lambda *a: None, now=lambda: self.sim.t, threads=False)
            self.assertEqual((len(freed), self.sd.streams[-1].closed), (2, True))

    def test_plays_and_reports_what_it_opened(self):
        out, player = self.output(0)
        for k in range(300):
            player.sb.feed(k * 480, self.sim.t + 0.6 + k * 0.01, vd.signal(k * 480, 480))
        self.sim.run(3)
        info = out.info()
        self.assertEqual(info["plan"]["blocksize"], 2048)
        self.assertTrue(info["plan"]["bt_class"])
        self.assertEqual(info["observed"]["frames_min"], 2048)
        self.assertEqual(info["observed"]["lead_ms"][1], 200.0)
        self.assertEqual((info["engine"], info["opened_reason"], info["call_profile"]), ("c-ring/2.0", "start", False))
        st = out.second()
        self.assertEqual((st["out"], st["io"], st["skips"], st["underruns"], st["overloads"]),
                         ("AirPods", [2048, 2048], 0, 0, 0))
        self.assertGreater(st["cb_per_s"], 20)
        self.assertIsNone(out.watchdog())

    def test_mono_device_is_a_call_profile(self):
        out, _ = self.output(4)
        self.assertTrue(out.call_profile)
        self.assertEqual(out.ring.channels, 1)


def ctypes_address(name):
    import ctypes
    return ctypes.cast(getattr(sacore.lib, name), ctypes.c_void_p).value


def wrap(fn, events, label):
    def f(*a, **k):
        events.append(label)
        return fn(*a, **k)
    return f


class LadderTest(OpenTest):
    """AudioOutput.watchdog(): when to reopen bigger, or elsewhere, or tell the user."""

    def feed(self, player, seconds):
        t0 = self.sim.t
        for k in range(int(seconds * 100)):
            player.sb.feed(k * 480, t0 + 0.6 + k * 0.01, vd.signal(k * 480, 480))

    def watch(self, out, seconds):
        actions = []
        self.sim.every(1.0, lambda: actions.append(out.watchdog()) if out.stream.active else None,
                       first=self.sim.t + 1.0)
        self.sim.advance(seconds)
        return [a for a in actions if a]

    def test_a_device_that_ignores_the_buffer_size_gets_level_1(self):
        out, player = self.output(3)            # asks for 1024, gets 256
        self.feed(player, 3)
        self.assertEqual(self.watch(out, 2.5)[0], {"action": "reopen", "level": 1, "from": 0,
                                                   "reason": "small_buffer", "persist": False})

    def test_macos_skipping_by_itself_goes_up_a_level_and_then_warns(self):
        out, player = self.output(1)
        self.feed(player, 20)
        self.sim.every(3.0, lambda dev=out.stream.device: dev.drop(1024), first=self.sim.t + 2.0)
        acts = self.watch(out, 12)
        self.assertEqual(acts[0], {"action": "reopen", "level": 1, "from": 0, "reason": "macos_skipping",
                                   "persist": True})
        self.outputs.remove(out)
        out.close()
        out, player = self.output(1, level=1)
        self.assertEqual(out.plan.blocksize, 2048)
        self.feed(player, 40)
        self.sim.every(3.0, lambda dev=out.stream.device: dev.drop(2048), first=self.sim.t + 2.0)
        acts = self.watch(out, 12)
        self.assertEqual(acts[0]["action"], "warn")
        self.assertEqual(acts[0]["warning"], "output_dropping")
        self.assertEqual(out.info()["warning"], "output_dropping")

    def test_callbacks_that_stall_twice_get_portaudios_own_size(self):
        out, player = self.output(1)
        self.feed(player, 10)
        dev = lambda: out.stream.device         # noqa: E731
        self.sim.at(self.sim.t + 2.0, lambda: dev().drop(int(0.4 * RATE)))
        self.sim.at(self.sim.t + 4.0, lambda: dev().drop(int(0.5 * RATE)))
        acts = self.watch(out, 6)
        self.assertEqual(acts[0], {"action": "reopen", "level": 2, "from": 0, "reason": "callbacks_stall",
                                   "persist": True})

    def test_a_device_that_stops_calling_is_gone(self):
        out, player = self.output(1)
        self.sim.at(self.sim.t + 2.0, out.stream.device.stop)
        acts = self.watch(out, 4)
        self.assertEqual(acts[0], {"action": "follow", "reason": "no_callbacks"})


class MacHelperTest(unittest.TestCase):
    """An output with macOS's helper (sacore.mac, faked here; otherwise only the Mac CI gets this far):
    the listeners it adds, and what their notifications make the watchdog do."""

    SEL = {"overload": 1, "latency": 2, "nominal_rate": 3, "alive": 4, "global": 10, "wildcard": 11, "output": 12}

    def setUp(self):
        self.sim = vd.Sim()
        self.sd = vd.VirtualSD(self.sim, [device("AirPods", 7680, lead=0.2)])
        self.facts = facts(7680, "blue", name="AirPods", uid="AirPods-uid")
        self.counters = {}
        names = {v: k for k, v in self.SEL.items()}
        test = self

        class Counter:
            value = 0

            def __init__(self, key):
                self.key = key

            def close(self):
                del test.counters[self.key]

        def listen(dev_id, sel, scope):
            key = (names[sel], names[scope])
            self.counters[key] = Counter(key)
            return self.counters[key]
        mac = types.SimpleNamespace(SEL=self.SEL, device_info=lambda *_: None, stream_device=lambda *_: 42,
                                    device_info_id=lambda *_: dict(self.facts), listen=listen)
        patch = mock.patch.object(sacore, "mac", mac)
        patch.start()
        self.addCleanup(patch.stop)

    def bump(self, what, scope, n=1):
        self.counters[what, scope].value += n

    def test_what_macos_says_about_an_output(self):
        out = ae.AudioOutput(self.sd, 0, RATE, Player(RATE, 2), SETTINGS, {}, log=lambda m: None,
                             now=lambda: self.sim.t, threads=False)
        self.sim.every(ae.OutputEngine.TICK, out.engine.tick, skip=lambda t: not out.stream.active)
        self.assertEqual(set(self.counters), {("overload", "global"), ("overload", "wildcard"), ("latency", "output"),
                                              ("nominal_rate", "global"), ("alive", "global")})
        self.sim.advance(3.0)
        self.assertIsNone(out.watchdog())
        self.bump("overload", "global", 2)
        self.bump("overload", "wildcard", 2)        # the wildcard listener hears the global ones too
        self.bump("latency", "output")
        st = out.second()
        self.assertEqual((st["uid"], st["overloads"], st["latency_changes"]), ("AirPods-uid", 2, 1))
        self.assertIsNone(out.watchdog())           # PortAudio follows a new latency itself
        self.bump("nominal_rate", "global")
        self.assertEqual(out.watchdog(), {"action": "reopen", "level": 0, "from": 0, "reason": "rate_changed",
                                          "persist": False})
        self.assertIsNone(out.watchdog())           # (said once)
        self.bump("alive", "global")                # still alive
        self.assertIsNone(out.watchdog())
        self.facts["alive"] = 0
        self.bump("alive", "global")
        self.assertEqual(out.watchdog(), {"action": "follow", "reason": "device_gone"})
        self.assertTrue(out.close())
        self.assertEqual(self.counters, {})         # every listener removed


class SelfTestTest(unittest.TestCase):
    """The real-time self-test on sacore's fake IO thread (short phases)."""

    @unittest.skipUnless(sacore.lib is not None and hasattr(sacore.lib, "sa_fake_start"), "needs the test build")
    def test_verdicts(self):
        for skip_every, want in ((0.0, "clean"), (0.5, "macos_dropping")):
            sd = vd.FakeSD([device("AirPods", 7680, lead=0.16, frames=64, skip_every=skip_every)])
            res = ae.SelfTest(sd, 0, RATE, log=lambda m: None, seconds=(0.6, 2.5)).run()
            self.assertEqual(res["legacy"]["frames_max"], 64)
            self.assertEqual(res["v2"]["io_frames"], 2048)
            self.assertEqual(res["v2"]["plan"]["blocksize"], 2048)
            self.assertEqual(res["v2"]["underruns"], 0, res)
            if skip_every:
                self.assertGreater(res["v2"]["skips"], 0, res)
                self.assertEqual(res["verdict"], want, res)
            else:                               # (the shared machine may still make the fake late)
                self.assertIn(res["verdict"], ("clean", "macos_dropping"), res)
                self.assertEqual(res["verdict"] == "clean", res["v2"]["skips"] == 0, res)
            self.assertEqual(sd.opened[0]["wrap_callback"], "array")    # the one Python callback: the old way
            self.assertIsInstance(sd.opened[1]["callback"], int)

    def test_verdict_rules(self):
        base = {"skips": 0, "overloads": 0, "lossy": False, "underruns": 0, "cb_max_us": 20}
        self.assertEqual(ae.verdict(base), "clean")
        self.assertEqual(ae.verdict({**base, "skips": 2}), "macos_dropping")
        self.assertEqual(ae.verdict({**base, "overloads": 1}), "macos_dropping")
        self.assertEqual(ae.verdict({**base, "lossy": True}), "macos_dropping")
        self.assertEqual(ae.verdict({**base, "underruns": 1}), "app_problem")
        self.assertEqual(ae.verdict({**base, "skips": 2, "cb_max_us": 5000}), "app_problem")


class ProbeTest(unittest.TestCase):
    def test_gc_timer(self):
        import gc
        t = ae.GcTimer().start()
        try:
            junk = [[i] for i in range(200000)]
            gc.collect()
            del junk
        finally:
            t.stop()
        st = t.second()
        self.assertGreater(st["gc_max_ms"], 0)
        self.assertGreaterEqual(st["gc_gen2"], 1)
        self.assertEqual(t.second()["gc_max_ms"], 0)

    def test_gil_probe_sees_a_held_gil(self):
        import ctypes
        import time
        p = ae.GilProbe().start()
        try:
            time.sleep(0.05)
            ctypes.PyDLL(None).usleep(60000)        # 60 ms with the GIL held
            time.sleep(0.05)
        finally:
            p.stop()
        self.assertGreater(p.second()["gil_probe_max_ms"], 40)


if __name__ == "__main__":
    unittest.main()
