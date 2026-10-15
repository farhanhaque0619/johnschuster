"""The ring of received audio and the resampler the output engine plays it through (native/sacore.c).

The engine plays content a hair faster or slower than its own sample rate to follow the device's
crystal, the other Mac's clock and delay glides, so the resampler must be inaudible: transparent
at speed 1, and clean (> 90 dB) up to 18 kHz at the speeds it uses.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sacore  # noqa: E402

RATE = 48000


class ResamplerTest(unittest.TestCase):
    def setUp(self):
        self.content = []

    def tearDown(self):
        for c in self.content:
            c.close()

    def make(self, channels=1):
        c = sacore.Content(channels, RATE)
        self.content.append(c)
        return c

    def test_the_filter_is_a_unit_impulse_on_whole_samples(self):
        f = sacore.FILTER
        self.assertEqual(f.shape, (257, 32))
        expect = np.zeros(32, np.float32)
        expect[15] = 1
        np.testing.assert_array_equal(f[0], expect)
        np.testing.assert_array_equal(f[256], np.roll(expect, 1))      # the next whole sample
        np.testing.assert_allclose(f.sum(axis=1), 1, atol=1e-6)        # no change in level

    def test_speed_one_on_a_whole_sample_is_bit_exact(self):
        c = self.make(2)
        x = (np.random.default_rng(0).standard_normal((48000, 2)) * 0.3).astype(np.float32)
        c.put(0, x)
        y = np.zeros((20000, 2), np.float32)
        self.assertEqual(c.resample(1000.0, 1.0, 20000, y), 0)
        np.testing.assert_array_equal(y, x[1000:21000])

    def test_clean_up_to_18_khz_at_the_speeds_used(self):
        c = self.make()
        n = np.arange(4 * RATE)
        for f in (1000, 10000, 18000):
            c.put(0, (0.5 * np.sin(2 * np.pi * f * n / RATE)).astype(np.float32)[:, None])
            for ratio in (1.0005, 0.997, 1.003):
                y = np.zeros((RATE, 1), np.float32)
                self.assertEqual(c.resample(1000.37, ratio, RATE, y), 0)
                exact = 0.5 * np.sin(2 * np.pi * f * (1000.37 + ratio * np.arange(RATE)) / RATE)
                snr = 10 * np.log10(np.mean(exact ** 2) / np.mean((y[:, 0] - exact) ** 2))
                self.assertGreater(snr, 90, (f, ratio))

    def test_a_gain_ramp_never_steps_more_than_its_slope(self):
        c = self.make(2)
        c.put(0, np.full((4000, 2), 0.5, np.float32))
        y = np.zeros((960, 2), np.float32)
        c.resample(1000.0, 1.0, 960, y, 0.0, 1.0)
        self.assertEqual((y[0, 0], y[-1, 0]), (0.0, 0.5))
        self.assertLessEqual(np.abs(np.diff(y[:, 0])).max(), 0.5 / 959 + 1e-6)     # float32 rounding
        # added onto what's there (a crossfade: one position fades out while another fades in)
        c.resample(2000.0, 1.0, 960, y, 1.0, 0.0, add=True)
        np.testing.assert_allclose(y, 0.5, atol=1e-6)

    def test_frames_without_content_are_counted_and_silent(self):
        c = self.make()
        y = np.ones((50, 1), np.float32)
        self.assertEqual(c.resample(0.0, 1.0, 50, y), 50)               # nothing received yet
        np.testing.assert_array_equal(y, 0)
        c.put(1000, np.ones((2000, 1), np.float32))
        self.assertEqual(c.resample(990.0, 1.0, 20, y), 10)             # 10 frames before it starts
        np.testing.assert_array_equal(y[10:20], 1)
        self.assertEqual(c.resample(2990.5, 1.0, 20, y), 10)            # and 10 after it ends
        for pos in (float("nan"), float("inf"), 1e300):                # nowhere: silence (without a NaN
            y.fill(1.0)                                                 # cast to an integer, which C leaves
            self.assertEqual(c.resample(pos, 1.0, 20, y), 20)           # undefined: R4's sanitizer checks)
            np.testing.assert_array_equal(y[:20], 0)
        self.assertEqual(c.resample(float("nan"), 1.0, 20, y, add=True), 20)
        np.testing.assert_array_equal(y[:20], 0)                       # (adding silence changes nothing)
        self.assertEqual(c.resample(1000.0, float("nan"), 20, y), 20)
        with self.assertRaises(ValueError):
            c.resample(1000.0, 1.0, 20, np.zeros((10, 1), np.float32))  # too short for n
        with self.assertRaises(ValueError):
            c.resample(1000.0, 1.0, 20, np.zeros((20, 1)))              # not float32


class ContentTest(unittest.TestCase):
    def test_holes_are_silence_and_counted(self):
        c = sacore.Content(2, RATE)
        self.assertEqual(c.range(), (None, None))
        self.assertEqual(c.put(0, np.ones((480, 2), np.float32)), 0)
        self.assertEqual(c.put(960, np.ones((480, 2), np.float32)), 480)   # a block never came
        self.assertEqual(c.range(), (0, 1440))
        y = np.zeros((1440, 2), np.float32)
        c.resample(0.0, 1.0, 1440, y)
        np.testing.assert_array_equal(y[480:960], 0)
        self.assertEqual(c.put(480, np.full((480, 2), 2, np.float32)), 0)  # ...then came late after all
        c.resample(0.0, 1.0, 1440, y)
        np.testing.assert_array_equal(y[480:960], 2)
        c.close()

    def test_the_oldest_audio_goes_and_a_far_jump_starts_over(self):
        c = sacore.Content(1, RATE)
        self.assertEqual(c.cap, 1 << 19)
        x = np.arange(960, dtype=np.float32)[:, None]
        n = 0
        while n < c.cap + 50000:
            c.put(n, x)
            n += 960
        self.assertEqual(c.range(), (n - c.cap + 4096, n))              # slots about to be reused: gone
        c.put(n + c.cap + 1, x)                                         # a new stream
        self.assertEqual(c.range(), (n + c.cap + 1, n + c.cap + 961))
        c.put(5, x)                                                     # and back again
        self.assertEqual(c.range(), (5, 965))
        with self.assertRaises(ValueError):
            c.put(0, np.zeros((c.cap, 1), np.float32))
        c.close()


if __name__ == "__main__":
    unittest.main()
