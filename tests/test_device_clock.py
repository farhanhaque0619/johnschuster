"""DeviceClock: the line from device frame index to host time, through PortAudio's wobbly stamps.

Pure arithmetic on made-up callback records (no audio, no threads): the device's true timeline is known
exactly, so the line's error is too.
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import audio_engine as ae  # noqa: E402

RATE = 48000


class Device:
    """True timeline: frame p reaches the ear at 1000 + (p + lost(p)) / true_rate. Stamps add `noise`."""

    def __init__(self, frames=1024, ppm=0.0, noise=lambda k: 0.0, lead=0.2):
        self.frames, self.true = frames, RATE * (1 + ppm * 1e-6)
        self.noise, self.lead = noise, lead
        self.k, self.lost = 0, 0.0

    def truth(self, pos):
        return 1000.0 + pos / self.true + self.lost

    def record(self, clock, stamp=None):
        pos = self.k * self.frames
        t = self.truth(pos)
        clock.add(pos, self.noise(self.k) + t if stamp is None else stamp, self.frames, t - self.lead)
        self.k += 1
        return pos

    @property
    def seconds(self):
        return self.k * self.frames / self.true


def ou(sigma, frames=1024, tau=1.0, seed=0):
    rng = np.random.default_rng(seed)
    a = math.exp(-frames / RATE / tau)
    x = [0.0]

    def next_value(k):
        x[0] = a * x[0] + math.sqrt(1 - a * a) * sigma * rng.normal()
        return x[0]
    return next_value


def run(dev, clock, seconds, ahead=0.25, judge_from=10.0, lose=None):
    """Feeds `seconds` of records; returns the line's errors predicting `ahead` s past each record."""
    errs = []
    while dev.seconds < seconds:
        if lose:
            lose(dev)
        pos = dev.record(clock)
        if dev.seconds > judge_from and clock.ready:
            p = pos + int(ahead * RATE)
            errs.append(clock.time_of(p) - dev.truth(p))
    return np.abs(errs) * 1000


def regular_losses(rate=0.007, start=0.0, seed=1, size=256, stop=math.inf):
    """`size`-frame slices the HAL drops (under the C side's half-buffer threshold), `rate` of the time."""
    rng = np.random.default_rng(seed)
    nxt = [None]

    def lose(dev):
        if not start <= dev.seconds < stop:
            return
        pos = dev.k * dev.frames
        if nxt[0] is None:
            nxt[0] = pos + size / rate * rng.uniform(0.5, 1.0)
        while pos >= nxt[0]:
            dev.lost += size / dev.true
            nxt[0] += size / rate * rng.uniform(0.7, 1.3)
    return lose


def lossy_from_to(dev, clock, lose, seconds):
    """When the clock first said lossy, and when it last did."""
    first = last = None
    while dev.seconds < seconds:
        lose(dev)
        dev.record(clock)
        if clock.lossy:
            first, last = first or dev.seconds, dev.seconds
    return first, last


class DeviceClockTest(unittest.TestCase):
    def test_white_jitter_of_2ms_is_filtered_out(self):
        rng = np.random.default_rng(0)
        e = run(Device(noise=lambda k: rng.uniform(-0.002, 0.002)), ae.DeviceClock(RATE), 40)
        self.assertLess(np.percentile(e, 99), 0.8)

    def test_wobble_that_wanders_stays_out_of_the_music(self):
        # Bluetooth's stamps wander +-2 ms with a 1 s memory (OU, sigma 1 ms): not followed
        for seed in range(5):
            clock = ae.DeviceClock(RATE)
            e = run(Device(noise=ou(0.001, seed=seed)), clock, 60)
            self.assertLess(e.max(), 3.0, seed)
            self.assertFalse(clock.lossy, seed)
            self.assertEqual(clock.reanchors, 0, seed)

    def test_follows_a_device_500_ppm_off_within_10_ppm_by_15s(self):
        for ppm in (500, -500):
            clock = ae.DeviceClock(RATE)
            dev = Device(ppm=ppm)
            run(dev, clock, 15)
            truth = (RATE / dev.true - 1) * 1e6         # rate_ppm: + = fewer frames per second
            self.assertLess(abs(clock.rate_ppm - truth), 10, (ppm, clock.rate_ppm))
            self.assertFalse(clock.lossy)

    def test_one_wild_report_is_ignored(self):
        clock, dev = ae.DeviceClock(RATE), Device()
        run(dev, clock, 12)
        dev.record(clock, stamp=dev.truth(dev.k * dev.frames) + 0.05)
        e = run(dev, clock, 14, judge_from=0)
        self.assertLess(e.max(), 0.001)
        self.assertEqual((clock.outliers, clock.reanchors), (1, 0))

    def test_three_in_a_row_restart_the_line_there(self):
        clock, dev = ae.DeviceClock(RATE), Device()
        run(dev, clock, 12)
        dev.lost -= 0.040                                   # the device's latency fell by 40 ms
        e = run(dev, clock, 14, judge_from=0)
        self.assertEqual((clock.outliers, clock.reanchors), (3, 1))
        self.assertLess(e[3:].max(), 0.001)

    def test_a_loss_too_small_to_see_is_flagged_lossy_within_8s(self):
        for start, seed in ((0.0, 1), (0.0, 2), (25.0, 3)):
            clock, dev = ae.DeviceClock(RATE), Device()
            lose = regular_losses(0.007, start, seed)
            at = None
            while dev.seconds < start + 20:
                lose(dev)
                dev.record(clock)
                if clock.lossy and at is None:
                    at = dev.seconds
            self.assertIsNotNone(at, (start, seed))
            self.assertLess(at - start, 8.0, (start, seed))
            self.assertAlmostEqual(clock.loss_pct, 0.7, delta=0.2)

    def test_losses_over_reanchor_but_under_a_skip_are_flagged_lossy_too(self):
        # AirPods' 2048-frame buffers: a lost 512-frame slice (10.7 ms) re-anchors the line like a latency
        # change would, but the C side doesn't see it (under 21 ms). Three of those in 30 s are losses
        first, _ = lossy_from_to(Device(frames=2048), ae.DeviceClock(RATE), regular_losses(0.007, size=512), 20)
        self.assertLess(first, 8.0)

    def test_losses_past_the_rate_clamp_are_flagged_lossy_too(self):
        # 2 %: more than the 1 % the narrow loop may learn, so the line re-anchors every second or so
        clock = ae.DeviceClock(RATE)
        first, _ = lossy_from_to(Device(), clock, regular_losses(0.02), 20)
        self.assertLess(first, 8.0)
        self.assertAlmostEqual(clock.loss_pct, 2.0, delta=0.4)

    def test_a_latency_step_under_a_skip_is_one_reanchor_not_a_loss(self):
        clock, dev = ae.DeviceClock(RATE), Device(frames=2048)
        run(dev, clock, 12)
        dev.lost += 0.015                                   # the device's latency grew by 15 ms
        e = run(dev, clock, 60, judge_from=0)
        self.assertEqual(clock.reanchors, 1)
        self.assertLess(e[3:].max(), 0.001)
        self.assertFalse(clock.lossy)

    def test_lossy_ends_once_30s_of_stamps_are_back_on_rate(self):
        clock, dev = ae.DeviceClock(RATE), Device()
        first, last = lossy_from_to(dev, clock, regular_losses(0.007, stop=20), 80)
        self.assertLess(first, 8.0)
        self.assertTrue(20 + 20 < last < 20 + 35, last)     # (the 30 s may start with a loss or two)
        self.assertEqual((clock.lossy, clock.loss_pct), (False, 0.0))

    def test_lossy_ends_on_a_device_whose_stamps_wobble(self):
        # Lossy's wide loop follows Bluetooth's wobble: its rate swings by thousands of ppm, and was
        # never under 500 for 30 s in a row. The stamps' slope over 30 s is
        for frames in (1024, 2048):
            for sigma in (0.001, 0.002):
                clock, dev = ae.DeviceClock(RATE), Device(frames=frames, noise=ou(sigma, frames))
                first, last = lossy_from_to(dev, clock, regular_losses(0.007, stop=20), 150)
                self.assertLess(first, 8.0, (frames, sigma))
                self.assertLess(last, 60, (frames, sigma))

    def test_missing_stamps_fall_back_to_now_plus_the_lead(self):
        # dev_time 0 (PortAudio had none): current_time + the median lead of the last good ones
        clock, dev = ae.DeviceClock(RATE), Device()
        run(dev, clock, 5)
        errs = []
        while dev.seconds < 10:
            pos = dev.k * dev.frames
            dev.record(clock, stamp=0.0)
            errs.append(abs(clock.time_of(pos) - dev.truth(pos)))
        self.assertLess(max(errs), 1e-6)
        self.assertGreater(clock.invalid, 200)

    def test_with_no_stamp_ever_the_streams_latency_stands_in(self):
        clock, dev = ae.DeviceClock(RATE, lead_guess=0.2), Device()
        for _ in range(10):
            dev.record(clock, stamp=0.0)
        self.assertTrue(clock.ready)
        self.assertAlmostEqual(clock.time_of(10 * 1024), dev.truth(10 * 1024), places=6)

    def test_stats(self):
        clock, dev = ae.DeviceClock(RATE, offset=0.004), Device(frames=2048)
        run(dev, clock, 3)
        st = clock.stats()
        self.assertEqual((st["frames_min"], st["frames_max"]), (2048, 2048))
        self.assertEqual(st["lead_ms"], [200.0, 200.0, 200.0])
        self.assertAlmostEqual(clock.time_of(0), 1000.004, places=6)        # + the stream latency
        self.assertEqual((st["lossy"], st["loss_pct"], st["reanchors"]), (False, 0.0, 0))


if __name__ == "__main__":
    unittest.main()
