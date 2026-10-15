"""Real-thread mechanism tests of the 2.0 audio core on Linux: the C callback runs on the tests'
fake IO thread (hard deadline, real-time priority where allowed), with Python on the other side
the way the app uses it, and what reached the "speaker" judged by true time (vdevice.judge).

R1  the canary: SyncAudio 1.1.6's Python callback under GIL stress must fail (or the test can't see
    the bug); 2.0 under the same stress, at 2048 and at 256 frames, must not
R2  2.0 with the GIL held for 200 ms every 2 s
R3  capture under GIL stress with input cycles skipped every 2 s
R4  everything that exercises libsacore, again under AddressSanitizer + UndefinedBehaviorSanitizer
R5  ring fuzz: random appends, new runs, rewrites and cuts against the running device

The fake device's own late wake-ups (the OS's fault, not the callback's) are counted apart: a
glitch they cause is not held against the engine.
"""
import contextlib
import ctypes
import importlib.util
import os
import re
import subprocess
import sys
import threading
import time
import types
import unittest

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
sys.path.insert(0, root)
sys.path.insert(0, here)
import sacore  # noqa: E402
import audio_engine as ae  # noqa: E402
import gil_stress  # noqa: E402
import vdevice as vd  # noqa: E402

RATE = 48000
F = 300                     # frames per device callback: not a divisor of the ring, so reads wrap too
GUARD = F + 64              # how close to the device the producer may write (the output engine's rule)
FADE = 64
SECONDS = float(os.environ.get("SACORE_TEST_SECONDS") or 30)
SECONDS_RT = min(SECONDS, 20.0)     # R1-R3
has_fake = sacore.lib is not None and hasattr(sacore.lib, "sa_fake_start")


def code(p, v):
    """What the fuzz writes at frame p in its v-th write, on channel 1; channel 0 is 1.0. A fade
    scales both channels alike, so the code still reads back exactly as channel 1 / channel 0."""
    return (((p & 0xFFFF) << 4) | (v & 0xF)) + 1


def stop_fake(fake):
    stats = (ctypes.c_uint64 * 6)()
    t0, maxcb = ctypes.c_double(), ctypes.c_double()
    sacore.lib.sa_fake_stop(fake, stats, ctypes.byref(t0), ctypes.byref(maxcb))
    return dict(zip(("cycles", "late", "skipped", "forced", "late_wake", "late_cb"), stats))


@unittest.skipUnless(has_fake, "needs the test build of libsacore")
class RingFuzzTest(unittest.TestCase):
    def test_random_appends_new_runs_rewrites_and_cuts_play_exactly(self):
        rng = np.random.default_rng(int(time.time()))
        ring = sacore.Ring("output", 2, RATE, seconds=8192 / RATE)     # 170 ms: wraps all the time
        total = int((SECONDS + 1) * RATE)
        fake = sacore.lib.sa_fake_start(0, ring.callback, ring.h, F, 2, RATE, 0.1, 0.0, 0.0, 0.0, total, 1)
        ops, logs, v = [], [], 0
        end = time.monotonic() + SECONDS
        stalls, next_stall = 0, time.monotonic() + 0.5
        try:
            while time.monotonic() < end:
                t0 = sacore.now()
                st = ring.state()
                r, w, s = st["r"], st["w"], st["s"]
                u = rng.random()
                if u > 0.97 and w > max(s, r + GUARD) + 1:                      # cut the end off
                    pos = int(rng.integers(max(s, r + GUARD), w))
                    if ring.cut(pos, GUARD):
                        ops.append((t0, sacore.now(), pos, 0, 0, r, w))
                    continue
                if w < r + GUARD or u < 0.05:
                    pos = max(w, r + GUARD) + 1 + int(rng.integers(0, 3000))    # a new run
                elif u < 0.35:
                    pos = int(rng.integers(r + GUARD, w))                       # rewrite, not yet played
                else:
                    pos = w                                                     # append
                n = min(int(rng.integers(1, 961)), r + ring.cap - pos + 300)    # sometimes too far: refused
                if n > 0:
                    v += 1
                    pcm = np.ones((n, 2), np.float32)
                    pcm[:, 1] = code(np.arange(pos, pos + n), v)
                    if ring.write(pos, pcm, GUARD) == n:
                        ops.append((t0, sacore.now(), pos, n, v, r, w))
                logs.append(ring.new_records())
                if w - r > ring.cap // 2:
                    time.sleep(0.002)                   # stay about 85 ms ahead, like the engine
                if time.monotonic() > next_stall:          # the producer stalls: shorter than its lead
                    stalls += 1                             # (inaudible), then longer (the ring runs dry)
                    time.sleep(rng.uniform(0.02, 0.06) if stalls % 2 else rng.uniform(0.12, 0.25))
                    next_stall = time.monotonic() + rng.uniform(0.2, 0.8)
        finally:
            stats = stop_fake(fake)
            rec = np.ctypeslib.as_array((ctypes.c_float * (total * 2)).from_address(
                sacore.lib.sa_fake_rec(fake))).reshape(total, 2).copy()
            sacore.lib.sa_fake_free(fake)
            logs.append(ring.new_records())
            st = ring.state()
            ring.close()
        logs = np.concatenate(logs)
        self.assertEqual(list(logs["seq"]), list(range(1, len(logs) + 1)), "a callback record went missing")
        bad, dropped, checked = self.judge(rec, logs, ops)
        summary = (f"{len(ops)} writes, {st['callbacks']} callbacks, {st['underrun_events']} underruns, "
                   f"{checked} frames checked, {dropped} buffers dropped by late callbacks, fake {stats}")
        print("\n  " + summary, file=sys.stderr)
        self.assertEqual(bad, [], summary)
        self.assertLessEqual(dropped, stats["late"], summary)
        self.assertGreater(st["underrun_events"], 0, "the fuzz never ran the ring dry")

    def judge(self, rec, logs, ops):
        """Replays the writes against the callback log: when each callback ran and which frames it
        played. A write that ended before the callback playing frame p started must show at p; one
        that overlapped it in time may or may not (only frame p's own index is checked there)."""
        total = len(rec)
        entry, exit_, cb = np.full(total, np.nan), np.full(total, np.nan), np.full(total, -1)
        for i, (pos, frames, t_in, t_out) in enumerate(zip(logs["pos"], logs["frames"],
                                                         logs["entry_time"], logs["exit_time"])):
            if pos + frames <= total:
                entry[pos:pos + frames], exit_[pos:pos + frames], cb[pos:pos + frames] = t_in, t_out, i
        expect = np.zeros(total, np.int64)          # which write's audio each frame plays (0: silence)
        loose = np.zeros(total, bool)
        s = ps = pe = 0                             # the current run starts at s; the one before was [ps, pe)

        def apply(lo, hi, value, t0, t1):
            hi = min(hi, total)
            if hi <= lo:
                return
            seg = slice(lo, hi)
            after = t0 > exit_[seg]                 # that frame had already played
            unsure = ~(t1 < entry[seg]) & ~after
            expect[seg] = np.where(after, expect[seg], value)
            loose[seg] |= unsure

        for t0, t1, pos, n, v, r, w in ops:
            if not n:                               # a cut: what was written from pos on won't play
                apply(pos, w, 0, t0, t1)
                continue
            if pos > w:                             # a new run: the old one still plays to its end, the
                apply(max(r, ps), pe, 0, t0, t1)    # one before it no longer does
                ps, pe, s = s, w, pos
            apply(max(pos, ps), min(pos + n, pe), v, t0, t1)
            apply(max(pos, s), pos + n, v, t0, t1)

        gain, ch1 = rec[:, 0].astype(np.float64), rec[:, 1].astype(np.float64)
        silent = (gain == 0) & (ch1 == 0)
        p = np.arange(total)
        # buffers the fake dropped because the callback returned late never reached the "speaker"
        played = cb >= 0
        heard = np.zeros(len(logs), bool)
        np.logical_or.at(heard, cb[played], ~silent[played])
        wanted = np.zeros(len(logs), bool)
        np.logical_or.at(wanted, cb[played], (expect[played] != 0) & ~loose[played])
        drop = wanted & ~heard
        check = played & ~drop[np.maximum(cb, 0)]
        # fades: a frame within FADE of silence may be scaled down
        near = np.convolve(silent, np.ones(2 * FADE + 1), "same") > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            got = np.where(silent, 0, np.rint(ch1 / gain)).astype(np.int64)
            exact_code = np.abs(ch1 / gain - got) < 0.25     # float32 rounding: < 0.13 at 2^20
        scaled_ok = (gain == 1) | ((gain > 0) & (gain < 1) & near)
        want = code(p, expect)
        strict_ok = np.where(expect == 0, silent, ~silent & exact_code & scaled_ok & (got == want))
        own_index = ((got - 1) >> 4) == (p & 0xFFFF)
        loose_ok = silent | (exact_code & scaled_ok & own_index)
        ok = np.where(loose, loose_ok, strict_ok)
        wrong = np.nonzero(check & ~ok)[0]
        bad = [(int(i), float(gain[i]), float(ch1[i]), int(expect[i]), bool(loose[i])) for i in wrong[:10]]
        return bad, int(drop.sum()), int(check.sum())


class Host(threading.Thread):
    """The host, in real time: 10 ms blocks of vdevice.signal, each handed over when it would have been
    captured, stamped T0 + n / RATE + D (this Mac's clock is the host's)."""

    def __init__(self, feed, T0, D):
        super().__init__(daemon=True)
        self.feed, self.T0, self.D = feed, T0, D
        self.done = threading.Event()

    def run(self):
        n = 0
        while not self.done.is_set():
            wait = self.T0 + (n + vd.BLOCK) / RATE - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.feed(n, self.T0 + n / RATE + self.D, vd.signal(n, vd.BLOCK))
            n += vd.BLOCK

    def stop(self):
        self.done.set()
        self.join(2)


def legacy116():
    """SyncAudio 1.1.6's engine, frozen (tests/legacy/syncaudio116.py = git show e08f2b1:syncaudio.py)."""
    spec = importlib.util.spec_from_file_location("syncaudio116", os.path.join(here, "legacy", "syncaudio116.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def airpods(frames, seconds, **kw):
    return {"name": "Fake AirPods", "max_output_channels": 2, "default_low_output_latency": 0.165,
            "lead": 0.2, "force_frames": frames, "record": seconds + 2, **kw}


def report(name, **values):
    print(f"\n  {name}: " + ", ".join(f"{k} {v}" for k, v in values.items()), file=sys.stderr)


@unittest.skipUnless(has_fake, "needs the test build of libsacore")
class StressTest(unittest.TestCase):
    D = 0.59                                # 2.0's start delay with AirPods (0.2 s of it to the ear)
    SETTINGS = types.SimpleNamespace(offset_ms=0.0, volume=1.0, channel="both")

    def setUp(self):
        self.switch = sys.getswitchinterval()
        sys.setswitchinterval(0.001)        # what the app sets: fairness for the producer thread

    def tearDown(self):
        sys.setswitchinterval(self.switch)

    def play_v2(self, device, seconds, stress=None, hog=None):
        """2.0 as the app opens it (AudioOutput, the producer thread) on the fake IO thread."""
        sd = vd.FakeSD([device])
        player = ae.LocalPlayer(RATE, 2)
        with gil_stress.Stress(stress) if stress else contextlib.nullcontext():
            out = ae.AudioOutput(sd, 0, RATE, player, self.SETTINGS, log=lambda m: None, reason="test")
            T0 = time.monotonic()
            host = Host(player.sb.feed, T0, self.D)
            host.start()
            if hog:
                hog.start()
            time.sleep(seconds)
            st = out.second()
            host.stop()
            fake = out.stream
            out.close()
        r = vd.judge(fake.rec, fake.tau(), vd.steady(T0, self.D), T0 + self.D + 2.0, T0 + seconds - 0.6)
        player.sb.content.close()
        return st, fake.stats, r

    def check_v2(self, frames, hog=False):
        held = threading.Event()

        def hold():                         # a C call that keeps the GIL for 200 ms, every 2 s
            usleep = ctypes.PyDLL(None).usleep
            while not held.wait(2.0):
                usleep(200_000)
        hogger = threading.Thread(target=hold, daemon=True) if hog else None
        try:
            st, fake, r = self.play_v2(airpods(frames, SECONDS_RT), SECONDS_RT, None if hog else "ci", hogger)
        finally:
            held.set()
        report(f"2.0 at {frames}" + (" with 200 ms GIL holds" if hog else " under ci stress"),
               glitches=r["glitches"], err_max_ms=r["err_max_ms"], underruns=st["underruns"], skips=st["skips"],
               ahead_ms=st["ahead_ms"], producer_max_ms=st["producer_max_ms"], **fake)
        self.assertEqual(st["underruns"], 0, (st, fake, r["glitch_at"]))
        self.assertLessEqual(r["glitches"], fake["late_wake"] + fake["forced"], (st, fake, r["glitch_at"]))
        if not hog:
            self.assertLessEqual(fake["late_cb"], 1, fake)
            self.assertLessEqual(r["err_max_ms"], 0.05, r["err_max_ms"])

    def test_r1_v2_at_2048_under_ci_stress(self):
        self.check_v2(2048)

    def test_r1_v2_at_256_under_ci_stress(self):
        self.check_v2(256)

    def test_r2_v2_rides_out_200ms_gil_holds(self):
        self.check_v2(1024, hog=True)

    def test_r1_legacy_116_fails_under_ci_stress(self):
        """The canary: the old engine on the same fake device under the same stress must show the bug,
        or this test can't see it."""
        legacy = legacy116()
        seconds = SECONDS_RT
        sd = vd.FakeSD([{**airpods(None, seconds), "force_frames": None, "frames": 2048}])
        player = legacy.Player(RATE, 2, lambda t: t)
        with gil_stress.Stress("ci"):
            out = legacy.SoundDeviceOutput(sd, 0, RATE, player.render)
            T0 = time.monotonic()
            host = Host(player.feed, T0, self.D)
            host.start()
            time.sleep(seconds)
            host.stop()
            out.close()
        fake = out.stream.stats
        lost = (fake["late"] + fake["skipped"]) * 2048 / RATE * 1000 / seconds
        r = vd.judge(out.stream.rec, out.stream.tau(), vd.steady(T0, self.D), T0 + self.D + 2.0, T0 + seconds - 0.6)
        report("1.1.6 at 2048 under ci stress (must fail)", lost_ms_per_s=round(lost, 1), glitches=r["glitches"],
               err_max_ms=r["err_max_ms"], **fake)
        self.assertTrue(fake["late_cb"] >= 5 or lost > 20, ("the canary didn't fail: the test can't see the bug", fake))

    def test_r3_capture_is_continuous_apart_from_zeroed_skips(self):
        sd = vd.FakeSD([{"name": "BlackHole 2ch", "max_input_channels": 2, "lead": 0.0, "skip_every": 2.0}])
        blocks = []
        with gil_stress.Stress("ci"):
            inp = ae.AudioInput(sd, 0, RATE, 2, lambda pcm, n, t: blocks.append((n, t, pcm[:, 0].copy())),
                                log=lambda m: None)
            time.sleep(SECONDS_RT * 0.6)
            inp.close()
        fake = inp.stream.stats
        n = np.concatenate([np.arange(b[0], b[0] + len(b[2])) for b in blocks])
        values = np.concatenate([b[2] for b in blocks])
        code = vd.decode(values)
        self.assertTrue(np.all(np.diff(n) == 1))                # n never jumps
        expect = n % vd.CODE                                    # the fake captures its own frame index
        zero = (values == 0) & (expect != vd.CODE // 2)
        self.assertTrue(np.all((code == expect) | zero), "a captured frame is in the wrong place")
        starts = np.flatnonzero(np.diff(np.r_[0, zero.astype(int)]) == 1)
        ends = np.flatnonzero(np.diff(np.r_[zero.astype(int), 0]) == -1)
        self.assertTrue(np.all((ends - starts + 1) % 512 == 0), "zeros that aren't whole skipped cycles")
        missing = fake["forced"] + fake["skipped"]
        report("capture under ci stress", blocks=len(blocks), zero_frames=int(zero.sum()), **fake)
        self.assertGreater(fake["forced"], 0)
        self.assertLessEqual(int(zero.sum()), missing * 512)    # (the last ones may not be read yet)
        self.assertGreaterEqual(int(zero.sum()), (missing - 1) * 512)
        line = inp.stream.t0 - 512 / RATE + n[::480] / RATE     # ADC time of frame k: t0 + k/RATE - a cycle
        stamps = np.array([b[1] for b in blocks])
        self.assertLess(np.abs(stamps - line).max(), 1e-5)


ASAN_TESTS = ["test_sacore_ring.RingTest", "test_sacore_ring.RealThreadTest", "test_resampler",
              "test_sacore.BindingTest", "test_engine_realtime.RingFuzzTest",
              "test_engine_realtime.StressTest.test_r1_v2_at_2048_under_ci_stress",
              "test_engine_realtime.StressTest.test_r1_v2_at_256_under_ci_stress",
              "test_output_engine.SteadyTest", "test_output_engine.DeviceTroubleTest", "test_output_engine.StallTest",
              "test_output_engine.DelayChangeTest.test_t9_step_up_is_the_same_silence_at_the_same_moment_on_both",
              "test_output_engine.DelayChangeTest.test_t9b_step_down_crossfades_exactly_when_the_new_line_is_due",
              "test_output_engine.DelayChangeTest.test_t9b_step_down_with_less_headroom_after_it_than_ahead",
              "test_capture", "test_plan.OpenTest", "test_plan.LadderTest", "test_coreaudio_ci.FakeLoopbackTest"]


class SanitizerTest(unittest.TestCase):
    """R4: libsacore built with ASan + UBSan (native/build.sh with SACORE_SANITIZE=1), loaded into
    a plain Python with the ASan runtime preloaded, running every test that drives it."""

    def test_no_memory_errors_or_undefined_behaviour(self):
        lib = os.path.join(root, "libsacore_asan.so")
        if not sys.platform.startswith("linux") or not os.path.exists(lib):
            self.skipTest("the sanitizer build is made on Linux (SACORE_SANITIZE=1 native/build.sh)")
        asan = subprocess.run(["cc", "-print-file-name=libasan.so"], capture_output=True, text=True).stdout.strip()
        if not os.path.isabs(asan):
            self.skipTest("the C compiler has no ASan runtime")
        env = {**os.environ, "SACORE_LIB": lib, "LD_PRELOAD": asan, "SACORE_TEST_SECONDS": "5",
               "ASAN_OPTIONS": "detect_leaks=0:halt_on_error=1",
               "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"}
        run = subprocess.run([sys.executable, "-m", "unittest"] + ASAN_TESTS, cwd=here, env=env,
                             capture_output=True, text=True, timeout=600)
        out = run.stdout + run.stderr
        self.assertIsNone(re.search(r"ERROR: AddressSanitizer|runtime error:", out), out[-4000:])
        self.assertEqual(run.returncode, 0, out[-4000:])
        self.assertRegex(out, r"Ran \d+ tests")


if __name__ == "__main__":
    unittest.main()
