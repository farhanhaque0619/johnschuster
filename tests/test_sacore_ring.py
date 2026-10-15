"""The rings CoreAudio's IO threads read and write (native/sacore.c).

Most tests play PortAudio's part by hand: they call a ring's C callback directly with a buffer and
made-up timestamps, so every frame and every counter can be checked exactly. The last ones run it
on real threads: the callback log under a writer that never pauses, a forced late callback on the
tests' fake IO thread, and a ThreadSanitizer build of the whole hand-over.
"""
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
import sacore  # noqa: E402

RATE = 48000
SECONDS = float(os.environ.get("SACORE_TEST_SECONDS") or 10)     # the long real-thread tests


class TimeInfo(ctypes.Structure):           # PaStreamCallbackTimeInfo
    _fields_ = [("inputBufferAdcTime", ctypes.c_double), ("currentTime", ctypes.c_double),
                ("outputBufferDacTime", ctypes.c_double)]


CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
                            ctypes.POINTER(TimeInfo), ctypes.c_ulong, ctypes.c_void_p)


def sig(pos, n):
    """Audio that says which frame it is: frame p is (p + 1, -(p + 1)), exact in float32."""
    p = np.arange(pos, pos + n, dtype=np.float64) + 1
    return np.stack([p, -p], 1).astype(np.float32)


def fade(x, up):
    """sacore.c's ramp(), in the same float32 arithmetic."""
    n = len(x)
    g = (np.arange(1, n + 1, dtype=np.float32) / np.float32(n + 1)).astype(np.float32)
    if not up:
        g = (np.float32(1) - g).astype(np.float32)
    return (x * g[:, None]).astype(np.float32)


class Device:
    """What PortAudio does with a ring: calls its callback once per buffer, with the DAC (output)
    or ADC (input) time of the buffer's first frame. `t` is when the next buffer is due."""

    def __init__(self, ring, frames=256, t=100.0):
        self.ring, self.frames, self.t = ring, frames, t
        self.latency = 0.0              # added to the timestamps only (the device's latency changed)
        self.cb = CALLBACK(ring.callback)

    def cycle(self, frames=None, t=None, flags=0, pcm=None, valid_time=True):
        n = frames or self.frames
        t = self.t if t is None else t
        stamp = t + self.latency if valid_time else 0.0
        out = self.ring.kind == "output"
        ti = TimeInfo(0.0 if out else stamp, t - 0.01, stamp if out else 0.0)
        if out:
            buf = np.full((n, self.ring.channels), np.nan, np.float32)   # every frame must be written
            self.cb(None, buf.ctypes.data, n, ctypes.byref(ti), flags, self.ring.h)
        else:
            buf = None
            self.cb(None if pcm is None else pcm.ctypes.data, None, n, ctypes.byref(ti), flags, self.ring.h)
        self.t = t + n / RATE
        return buf

    def play(self, buffers):
        return np.concatenate([self.cycle() for _ in range(buffers)])

    def skip(self, frames):
        """macOS skips IO cycles: time moves on, the callback isn't called."""
        self.t += frames / RATE


def ring(kind="output", cap=4096, channels=2):
    r = sacore.Ring(kind, channels, RATE, seconds=cap / RATE)
    assert r.cap == cap
    return r


class RingTest(unittest.TestCase):
    def setUp(self):
        self.rings = []

    def tearDown(self):
        for r in self.rings:
            r.close()

    def ring(self, *a, **k):
        r = ring(*a, **k)
        self.rings.append(r)
        return r

    def test_version_and_clock(self):
        self.assertEqual(sacore.VERSION, 0x020000)
        self.assertLess(abs(sacore.now() - time.monotonic()), 1e-3)

    def test_audio_comes_out_exactly_across_the_wrap(self):
        r = self.ring()
        dev = Device(r, frames=300)                 # not a divisor of the ring: reads straddle the end too
        out = []
        w = 0
        for _ in range(48):                     # 3 times round the ring, writes and reads both wrapping
            while w + 1000 <= r.state()["r"] + r.cap:
                self.assertEqual(r.write(w, sig(w, 1000), 0), 1000)
                w += 1000
            out.append(dev.cycle())
        out = np.concatenate(out)
        np.testing.assert_array_equal(out, sig(0, len(out)))
        st = r.state()
        self.assertEqual((st["r"], st["callbacks"], st["underrun_events"], st["skips"]), (len(out), 48, 0, 0))

    def test_capture_comes_out_exactly_across_the_wrap(self):
        r = self.ring("input")
        dev = Device(r, frames=300)
        got, pos = [], 0
        for k in range(48):
            dev.cycle(pcm=sig(k * 300, 300))
            while (x := r.read(pos, 480)) is not None:
                got.append(x)
                pos += 480
        got = np.concatenate(got)
        np.testing.assert_array_equal(got, sig(0, len(got)))
        self.assertEqual((len(got), r.state()["overflows"]), (48 * 300 // 480 * 480, 0))

    def test_writes_too_close_or_too_far_ahead_are_refused(self):
        r = self.ring()
        dev = Device(r)
        dev.play(2)                                                     # r = 512
        self.assertEqual(r.write(512 + 299, sig(0, 10), 300), -1)       # inside the guard
        self.assertEqual(r.write(512 + 300, sig(0, 10), 300), 10)
        self.assertEqual(r.write(512 + r.cap - 9, sig(0, 10), 0), -1)   # a ring ahead: would overwrite r
        self.assertEqual(r.write(512 + r.cap - 10, sig(0, 10), 0), 10)
        with self.assertRaises(ValueError):
            r.write(2000, np.zeros((10, 1), np.float32), 0)             # wrong channel count

    def test_append_rewrite_and_new_run(self):
        r = self.ring()
        dev = Device(r)
        r.write(0, sig(0, 1000), 0)
        r.write(1000, sig(1000, 1000), 0)                               # append
        st = r.state()
        self.assertEqual((st["w"], st["s"]), (2000, 0))
        new = sig(1500, 200) * 7
        r.write(1500, new, 0)                                           # rewrite, not yet played
        self.assertEqual((r.state()["w"], r.state()["s"]), (2000, 0))
        out = dev.play(4)                                               # [0, 1024)
        np.testing.assert_array_equal(out, sig(0, 1024))
        r.write(3000, sig(3000, 500), 0)                                # a new run: s moves first
        st = r.state()
        self.assertEqual((st["w"], st["s"]), (3500, 3000))
        out = dev.play(10)                                              # [1024, 3584)
        # the old run plays to its end and fades out, then silence until the new run (not an
        # underrun: the silence between runs isn't missing audio), which fades in where it starts
        old = sig(1024, 2000 - 1024)
        old[1500 - 1024:1700 - 1024] *= 7
        old[-64:] = fade(old[-64:], False)
        np.testing.assert_array_equal(out[:2000 - 1024], old)
        np.testing.assert_array_equal(out[2000 - 1024:3000 - 1024], 0)
        np.testing.assert_array_equal(out[3000 - 1024:3064 - 1024], fade(sig(3000, 64), True))
        np.testing.assert_array_equal(out[3064 - 1024:3500 - 64 - 1024], sig(3064, 3500 - 64 - 3064))
        np.testing.assert_array_equal(out[3500 - 64 - 1024:3500 - 1024], fade(sig(3436, 64), False))
        np.testing.assert_array_equal(out[3500 - 1024:], 0)
        st = r.state()
        self.assertEqual((st["underrun_events"], st["underrun_frames"]), (1, 3584 - 3500))

    def test_a_new_run_while_the_ring_is_low_lets_the_old_one_finish(self):
        """The producer starts a new run when the ring is nearly empty (it can't write that close to
        the device): what is left of the old run still plays and fades out, wherever it ends."""
        r = self.ring()
        dev = Device(r)
        r.write(0, sig(0, 1000), 0)
        dev.play(3)                                                     # r 768: 232 frames left
        self.assertEqual(r.write(768 + 300 + 100, sig(1168, 500), 300), 500)
        out = dev.play(2)                                               # [768, 1280)
        np.testing.assert_array_equal(out[:1000 - 64 - 768], sig(768, 1000 - 64 - 768))
        np.testing.assert_array_equal(out[1000 - 64 - 768:1000 - 768], fade(sig(936, 64), False))
        np.testing.assert_array_equal(out[1000 - 768:1168 - 768], 0)
        np.testing.assert_array_equal(out[1168 - 768:1232 - 768], fade(sig(1168, 64), True))
        self.assertEqual(r.state()["underrun_events"], 0)
        # an old run ending exactly where a buffer ends fades out too: nothing can follow it now
        r2 = self.ring()
        dev = Device(r2)
        r2.write(0, sig(0, 1024), 0)
        dev.play(3)
        r2.write(1500, sig(1500, 100), 300)
        out = dev.cycle()                                               # [768, 1024)
        np.testing.assert_array_equal(out[-64:], fade(sig(960, 64), False))
        np.testing.assert_array_equal(out[:-64], sig(768, 192))

    def test_cut_ends_the_ring_where_the_producer_stopped(self):
        r = self.ring()
        dev = Device(r)
        r.write(0, sig(0, 2000), 0)
        dev.play(1)                                                     # r 256
        self.assertFalse(r.cut(256 + 299, 300))                         # inside the guard
        self.assertFalse(r.cut(2000, 300))                              # not before w
        self.assertTrue(r.cut(1500, 300))
        self.assertEqual(r.state()["w"], 1500)
        r.write(1500, sig(1500, 100) * 7, 300)                          # what is written next appends there
        out = dev.play(6)                                               # [256, 1792)
        np.testing.assert_array_equal(out[:1600 - 64 - 256], sig(256, 1600 - 64 - 256) *
                                      np.r_[np.ones(1500 - 256), np.full(100 - 64, 7)][:, None])
        np.testing.assert_array_equal(out[1600 - 256:], 0)              # the old audio after it never plays
        r.write(2500, sig(2500, 100), 300)                              # a new run...
        self.assertFalse(r.cut(2400, 300))                              # ...can't be cut before its start

    def test_rewritten_audio_plays(self):
        r = self.ring()
        dev = Device(r)
        r.write(0, sig(0, 2048), 0)
        r.write(1500, sig(1500, 200) * 7, 0)
        out = dev.play(8)
        expect = sig(0, 2048)
        expect[1500:1700] *= 7
        np.testing.assert_array_equal(out, expect)

    def test_underruns_count_once_per_gap_and_only_after_the_first_write(self):
        r = self.ring()
        dev = Device(r)
        dev.play(4)                                                     # nothing written yet: not an underrun
        self.assertEqual(r.state()["underrun_events"], 0)
        r.write(1024 + 100, sig(0, 300), 0)                             # a run [1124, 1424)
        dev.play(6)                                                     # [1024, 2560): runs dry at 1424
        st = r.state()
        self.assertEqual((st["underrun_events"], st["underrun_frames"]), (1, 2560 - 1424))
        r.write(2560 + 256, sig(0, 256), 0)                             # a fresh run starts exactly on a buffer
        dev.play(4)                                                     # [2560, 3584)
        st = r.state()
        self.assertEqual((st["underrun_events"], st["underrun_frames"]), (2, 2560 - 1424 + 3584 - 3072))

    def test_skips_are_detected_from_half_a_buffer(self):
        r = self.ring(cap=1 << 15)
        dev = Device(r, frames=1024)
        r.write(0, sig(0, 30000), 0)
        dev.play(2)
        dev.skip(0.49 * 1024)                                           # report wobble: not a skip
        dev.cycle()
        self.assertEqual(r.state()["skips"], 0)
        dev.skip(0.51 * 1024)                                           # a skipped cycle's worth
        dev.cycle()
        st = r.state()
        self.assertEqual((st["skips"], st["skipped_frames"], st["r"]), (1, 522, 4 * 1024 + 522))
        rec = r.new_records()[-1]
        self.assertEqual((rec["pos"], rec["skipped"], rec["frames"]), (3 * 1024 + 522, 522, 1024))
        dev.skip(2.01 * RATE)                                           # 2 s or more: not a skip...
        dev.cycle()
        dev.cycle()                                                     # ...and the line restarts there
        self.assertEqual(r.state()["skips"], 1)

    def test_skipped_cycles_are_whole_buffers_whatever_the_wobble(self):
        """Bluetooth timestamps wobble by a few ms: a skipped cycle still moves the index by exactly
        its buffer, so the index -> time line stays where it was."""
        r = self.ring(cap=1 << 16)
        dev = Device(r, frames=2048)
        r.write(0, sig(0, 60000), 0)
        dev.play(2)
        for skip, wobble in ((1, 0.0019), (1, -0.0019), (2, 0.0015)):   # up to ~91 frames off
            dev.skip(skip * 2048)
            dev.cycle(t=dev.t + wobble)
            dev.t -= wobble                                             # (the wobble isn't time)
            dev.cycle()
        st = r.state()
        self.assertEqual((st["skips"], st["skipped_frames"]), (3, 4 * 2048))
        dev.skip(0.7 * 2048)                                            # not whole cycles: as measured
        dev.cycle()
        self.assertEqual(r.state()["skipped_frames"], 4 * 2048 + round(0.7 * 2048))

    def test_a_latency_step_is_skipped_as_measured_even_near_whole_buffers(self):
        """The device's latency grew (Bluetooth renegotiated): only the timestamps jump, not when the IO
        cycles come, so the jump is exactly what to skip - 40 ms is 0.94 of a 2048-frame buffer, but not
        a buffer."""
        r = self.ring(cap=1 << 16)
        dev = Device(r, frames=2048)
        r.write(0, sig(0, 60000), 0)
        dev.play(2)
        dev.latency = 0.040
        dev.play(2)
        st = r.state()
        self.assertEqual((st["skips"], st["skipped_frames"]), (1, round(0.040 * RATE)))

    def test_a_missing_timestamp_is_not_a_skip(self):
        r = self.ring(cap=1 << 15)
        dev = Device(r, frames=470)
        r.write(0, sig(0, 30000), 0)
        for frames in (470, 471, 470, 471):                             # blocksize 0: sizes vary
            dev.cycle(frames)
        dev.cycle(471, valid_time=False)                                # PortAudio had no time for this one
        dev.cycle(470)
        dev.cycle(471, valid_time=False)
        dev.cycle(470, valid_time=False)
        dev.skip(1024)
        dev.cycle(471)                                                  # a real skip, after two unknowns
        st = r.state()
        self.assertEqual((st["skips"], st["skipped_frames"]), (1, 1024))
        recs = r.new_records()
        self.assertEqual(list(recs["dev_time"] > 0), [True] * 4 + [False, True, False, False, True])

    def test_fades_at_every_edge(self):
        r = self.ring()
        dev = Device(r)
        r.write(256, sig(256, 256), 0)                                  # a run starting on a buffer edge
        out = dev.play(2)
        np.testing.assert_array_equal(out[256:], sig(256, 256))        # after silence before s: no fade
        # missing audio, then audio from the start of a buffer: fades in
        dev.play(1)                                                     # [512, 768) missing
        r.write(768, sig(768, 512), 0)
        out = dev.play(2)
        np.testing.assert_array_equal(out[:64], fade(sig(768, 64), True))
        np.testing.assert_array_equal(out[64:], sig(832, 448))
        # macOS skipped a cycle: what comes after it fades in
        r.write(1280, sig(1280, 1024), 0)
        dev.skip(256)
        out = dev.cycle()                                               # [1536, 1792)
        np.testing.assert_array_equal(out[:64], fade(sig(1536, 64), True))
        np.testing.assert_array_equal(out[64:], sig(1600, 192))
        # a run shorter than the fade fades in and out over its whole length
        dev.play(2)                                                     # to 2304, where that write ends
        r.write(2560 + 100, sig(0, 40), 0)
        np.testing.assert_array_equal(dev.cycle(), 0)                   # [2304, 2560): before the run
        out = dev.cycle()                                               # [2560, 2816)
        np.testing.assert_array_equal(out[100:140], fade(fade(sig(0, 40), True), False))
        np.testing.assert_array_equal(np.delete(out, np.s_[100:140], 0), 0)

    def test_flags_callback_cost_and_lead_are_since_the_last_read(self):
        r = self.ring()
        dev = Device(r)
        self.assertEqual(r.state()["min_lead_frames"], sacore.NO_LEAD)
        r.write(0, sig(0, 2000), 0)
        dev.cycle(flags=0x4)                                            # paOutputUnderflow
        dev.cycle(flags=0x10)                                           # paPrimingOutput
        st = r.state()
        self.assertEqual((st["flag_or"], st["min_lead_frames"]), (0x14, 2000 - 256))
        self.assertGreater(st["max_cb_ns"], 0)
        self.assertLess(st["max_cb_ns"], 3e6)
        st = r.state()
        self.assertEqual((st["flag_or"], st["max_cb_ns"], st["min_lead_frames"]), (0, 0, sacore.NO_LEAD))
        r.debug_stall(3000)                                             # CI's forced overload: once
        dev.cycle()
        dev.cycle()
        self.assertGreaterEqual(r.state()["max_cb_ns"], 3e6)
        recs = r.new_records()
        cost = recs["exit_time"] - recs["entry_time"]
        self.assertGreaterEqual(cost[-2], 3e-3)
        self.assertLess(cost[-1], 3e-3)                                 # once only

    def test_the_log_says_what_each_callback_did(self):
        r = self.ring()
        dev = Device(r)
        r.write(100, sig(0, 300), 0)
        t0 = dev.t
        before = sacore.now()
        dev.play(3)
        recs = r.new_records()
        self.assertEqual(list(recs["seq"]), [1, 2, 3])
        self.assertEqual(list(recs["pos"]), [0, 256, 512])
        self.assertEqual(list(recs["valid"]), [156, 144, 0])
        self.assertEqual(list(recs["frames"]), [256] * 3)
        np.testing.assert_allclose(recs["dev_time"], t0 + np.array([0, 256, 512]) / RATE, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(recs["current_time"], recs["dev_time"] - 0.01)
        self.assertTrue(np.all((before <= recs["entry_time"]) & (recs["entry_time"] <= recs["exit_time"])))
        self.assertEqual(len(r.new_records()), 0)                       # each record once
        dev.play(5000)
        recs = r.new_records()                                          # fell behind: the newest LOG_N
        self.assertEqual((len(recs), recs["seq"][0], recs["seq"][-1]), (sacore.LOG_N, 5003 - 4095, 5003))

    def test_input_skips_are_silence_at_their_own_frames(self):
        r = self.ring("input", cap=8192)
        dev = Device(r, frames=512)
        for k in range(8):
            if k == 5:
                dev.skip(1024)                                          # two cycles never arrived
            pos = round((dev.t - 100.0) * RATE)
            dev.cycle(pcm=None if k == 7 else sig(pos, 512))            # the last one without data
        st = r.state()
        self.assertEqual((st["w"], st["skips"], st["skipped_frames"]), (8 * 512 + 1024, 1, 1024))
        recs = r.new_records()
        self.assertEqual((recs["pos"][5], recs["skipped"][5]), (5 * 512 + 1024, 1024))
        expect = sig(0, 8 * 512 + 1024)
        expect[5 * 512:5 * 512 + 1024] = 0
        expect[-512:] = 0
        np.testing.assert_array_equal(r.read(0, 8 * 512 + 1024), expect)

    def test_input_read_waits_and_reports_lapping(self):
        r = self.ring("input")
        dev = Device(r, frames=512)
        for _ in range(4):
            dev.cycle(pcm=sig(0, 512))
        self.assertIsNone(r.read(1500, 600))                            # not all captured yet
        self.assertEqual(len(r.read(1500, 548)), 548)
        for _ in range(10):                                             # 5120 more: past a whole ring
            dev.cycle(pcm=sig(0, 512))
        with self.assertRaises(sacore.Lapped):
            r.read(2048, 512)
        self.assertEqual(len(r.read(r.state()["w"] - 512, 512)), 512)
        self.assertEqual(r.state()["overflows"], 2)                     # behind by a ring in 2 callbacks


@unittest.skipUnless(sacore.lib is not None and hasattr(sacore.lib, "sa_fake_start"), "needs the test build")
class RealThreadTest(unittest.TestCase):
    def test_the_log_is_never_torn_while_written(self):
        # The fake IO thread at 1 frame per cycle of a "1 GHz" device can never keep up, so it calls
        # back without pausing (a log record every few hundred ns, skips everywhere) while this thread
        # reads the oldest records - the slots being overwritten - as fast as it can.
        rate, lead = 1e9, 0.5
        r = sacore.Ring("output", 1, rate, seconds=1e-5)
        fake = sacore.lib.sa_fake_start(0, r.callback, r.h, 1, 1, rate, lead, 0.0, 0.0, 0.0, 0, 0)
        buf = np.zeros(sacore.LOG_N, sacore.LOG_DTYPE)
        accepted = raced = 0
        end = time.monotonic() + SECONDS
        try:
            while time.monotonic() < end:
                last = r.state()["log_seq"]
                n = sacore.lib.sa_ring_log(r.h, max(0, last - sacore.LOG_N), buf.ctypes.data, sacore.LOG_N)
                raced += n < sacore.LOG_N
                accepted += n
                self.check(buf[:n], rate, lead)
        finally:
            stats = (ctypes.c_uint64 * 6)()
            t0, maxcb = ctypes.c_double(), ctypes.c_double()
            sacore.lib.sa_fake_stop(fake, stats, ctypes.byref(t0), ctypes.byref(maxcb))
            sacore.lib.sa_fake_free(fake)
            r.close()
        self.assertGreater(stats[0], 1e6)
        print(f"\n  {stats[0]} callbacks, {accepted} records read, {raced} reads met records being rewritten",
              file=sys.stderr)

    def check(self, recs, rate, lead):
        if not len(recs):
            return
        self.assertTrue(np.all(np.diff(recs["seq"].astype(np.int64)) > 0), "records out of order")
        self.assertTrue(np.all(recs["frames"] == 1), "torn: frames")
        self.assertTrue(np.all(recs["exit_time"] >= recs["entry_time"]), "torn: times")
        self.assertTrue(np.all(np.diff(recs["pos"].astype(np.int64)) >= np.diff(recs["seq"].astype(np.int64))),
                        "torn: positions")
        # frame k reaches the "DAC" at t0 + lead + k / rate, and the ring follows skips exactly
        t0 = recs["dev_time"][0] - lead - recs["pos"][0] / rate
        k = np.rint((recs["dev_time"] - lead - t0) * rate)
        np.testing.assert_array_equal(k, recs["pos"].astype(np.float64), "torn: time vs position")

    def test_a_late_callback_is_one_counted_skip_and_costs_no_time(self):
        # The premise of the whole design (checked on a real Mac in CI P4): a callback that is too
        # late makes the device skip cycles, and the next timestamp jumps by exactly what was skipped.
        r = sacore.Ring("output", 2, RATE)
        r.write(0, np.full((RATE, 2), 0.1, np.float32), 0)
        fake = sacore.lib.sa_fake_start(0, r.callback, r.h, 1024, 2, RATE, 0.1, 0.0, 0.0, 0.0, 0, 1)
        time.sleep(0.3)
        r.debug_stall(60000)
        time.sleep(0.5)
        stats = (ctypes.c_uint64 * 6)()
        t0, maxcb = ctypes.c_double(), ctypes.c_double()
        sacore.lib.sa_fake_stop(fake, stats, ctypes.byref(t0), ctypes.byref(maxcb))
        sacore.lib.sa_fake_free(fake)
        cycles, late, skipped, forced, late_wake, late_cb = stats
        st = r.state()
        r.close()
        self.assertGreaterEqual(late_cb, 1, stats[:])
        self.assertGreaterEqual(skipped, 2, stats[:])                   # 60 ms at 21 ms per cycle
        self.assertEqual(st["skipped_frames"], skipped * 1024, (st, stats[:]))
        if late == 1:
            self.assertEqual(st["skips"], 1)
        self.assertEqual(st["r"], (cycles + skipped) * 1024)            # index == device time, still


class LapTest(unittest.TestCase):
    def test_a_reader_being_lapped_never_gets_overwritten_audio(self):
        """Capture fell a whole ring behind and the IO thread is overwriting what it reads, right now:
        the read is either right or refused (sacore_lap.c, a plain optimized build)."""
        cc = next(filter(None, map(shutil.which, ("cc", "clang", "gcc"))), None)
        if cc is None:
            self.skipTest("no C compiler")
        tmp = tempfile.mkdtemp()
        exe = os.path.join(tmp, "lap")
        frameworks = ["-framework", "CoreAudio", "-framework", "CoreFoundation"] if sys.platform == "darwin" else []
        built = subprocess.run([cc, "-std=c11", "-O2", "-o", exe, os.path.join(here, "sacore_lap.c"), "-lpthread", "-lm"]
                               + frameworks, capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stderr[-2000:])
        run = subprocess.run([exe, str(min(SECONDS, 2))], capture_output=True, text=True, timeout=60)
        shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr[-2000:])
        print(f"\n  {run.stdout.strip()}", file=sys.stderr)


class TSanTest(unittest.TestCase):
    def test_the_hand_overs_are_race_free_under_threadsanitizer(self):
        tmp = tempfile.mkdtemp()
        exe = os.path.join(tmp, "race")
        frameworks = ["-framework", "CoreAudio", "-framework", "CoreFoundation"] if sys.platform == "darwin" else []
        errors = []
        for cc in filter(None, map(shutil.which, ("cc", "clang", "gcc"))):    # any that has TSan's runtime
            built = subprocess.run([cc, "-std=c11", "-O1", "-g", "-fsanitize=thread", "-o", exe,
                                    os.path.join(here, "sacore_race.c"), "-lpthread", "-lm"] + frameworks,
                                   capture_output=True, text=True)
            if not built.returncode:
                break
            errors.append(f"{cc}: {built.stderr.strip()[-200:]}")
        else:
            self.skipTest("no compiler here can build with ThreadSanitizer: " + " | ".join(errors))
        env = {**os.environ, "TSAN_OPTIONS": "halt_on_error=0 exitcode=66"}
        env.pop("LD_PRELOAD", None)
        run = subprocess.run([exe, str(min(SECONDS, 4))], capture_output=True, text=True, env=env, timeout=120)
        shutil.rmtree(tmp, ignore_errors=True)
        if "FATAL: ThreadSanitizer" in run.stderr:          # e.g. a kernel mapping layout TSan can't use
            self.skipTest(run.stderr.strip()[-300:])
        self.assertNotIn("ThreadSanitizer", run.stderr, run.stderr[-3000:])
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr[-3000:])
        print(f"\n  {run.stdout.strip()}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
