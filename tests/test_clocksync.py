"""ClockSync: this Mac's clock onto the host's, from UDP pings over noisy, lopsided Wi-Fi - and the
applied mapping the players use, which moves at most 200 µs a second and jumps only when the
estimate stays far away."""
import os
import sys
import unittest

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
import syncaudio as sa  # noqa: E402


def pings(clock, seconds, local, rng, start=0.0, every=0.2, shift=lambda T: 0.0):
    """Feed `seconds` of pings from host time `start`; the host's clock is T + shift(T) (a step in it
    is the estimate moving). Returns [(T, applied mapping's error, its slew at that estimate)]."""
    out, T = [], start
    while T < start + seconds:
        up, down = 0.0015 + rng.exponential(0.003), 0.001 + rng.exponential(0.002)
        t1, t2 = local(T), T + up + shift(T)
        t3 = t2 + 0.0001
        t4 = local(T + up + 0.0001 + down)
        clock.add(t1, t2, t3, t4)
        T += up + 0.0001 + down + every
        if clock.estimate:
            out.append((T, clock.to_server(local(T)) - (T + shift(T)), clock.slew))
    return out


class ClockSyncTest(unittest.TestCase):
    def test_tracks_offset_and_drift_through_lopsided_wifi(self):
        rng = np.random.default_rng(3)
        offset, drift = 1234.5, 80e-6           # client clock = offset + T * (1 + drift)
        local = lambda T: offset + T * (1 + drift)      # noqa: E731
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
        self.assertAlmostEqual(clock.server_rate(), 1 / (1 + drift), delta=20e-6)

    def test_unsynced_until_enough_pings(self):
        clock = sa.ClockSync()
        self.assertIsNone(clock.to_server(1.0))
        self.assertEqual(clock.server_rate(), 1.0)
        for i in range(sa.ClockSync.MIN_SAMPLES):
            clock.add(i, i + 0.001, i + 0.001, i + 0.002)
        self.assertAlmostEqual(clock.to_server(100.0), 100.0, places=6)

    def settled(self, seed=1):
        """A clock synced for 40 s (past its first 30)."""
        rng = np.random.default_rng(seed)
        local = lambda T: 500.0 + T             # noqa: E731
        clock = sa.ClockSync()
        pings(clock, 40, local, rng)
        return clock, local, rng

    def test_a_raw_offset_step_of_10_ms_is_followed_at_200_us_a_second(self):
        clock = sa.ClockSync()
        for i in range(20):                                     # settled at offset 0, drift 0
            clock.add(i, i + 0.001, i + 0.001, i + 0.002)
        moved = []
        for k in range(300):                                    # the estimate says 10 ms from now on
            t = 20.0 + 0.2 * k
            clock.raw = (t, 0.010, 0.0)
            clock._apply(t)
            moved.append(clock.estimate[1])
        self.assertTrue(np.allclose(np.diff(moved)[:200], 200e-6 * 0.2))       # exactly 200 us a second
        self.assertAlmostEqual(moved[-1], 0.010)                                # there after 50 s
        self.assertEqual(clock.steps, 0)

    def test_a_10_ms_move_on_the_wire_is_followed_slowly(self):
        """The host's clock seems to move 10 ms (the Wi-Fi route changed): the estimate's line fit runs
        off to its 500 ppm limit for a while; the applied mapping still only creeps."""
        clock, local, rng = self.settled()
        trace = np.array(pings(clock, 80, local, rng, start=40.0, shift=lambda T: 0.010))
        self.assertLessEqual(np.abs(trace[:, 2]).max(), 200e-6 + 1e-9)     # each estimate: 200 us a second
        moving = np.abs(np.diff(trace[:, 1]) / np.diff(trace[:, 0]))
        self.assertLess(moving.max(), 400e-6)                              # in all: under 0.04 % speed
        self.assertLess(np.abs(trace[-50:, 1]).max(), 0.001)               # and there in the end
        self.assertEqual(clock.steps, 0)

    def test_25_ms_for_3_estimates_is_one_jump(self):
        clock, local, rng = self.settled()
        trace = np.array(pings(clock, 60, local, rng, start=40.0, shift=lambda T: 0.025))
        self.assertEqual(clock.steps, 1)
        self.assertAlmostEqual(clock.last_step, 0.023, delta=0.004)
        self.assertLess(abs(trace[-1, 1]), 0.0015)

    def test_drift_is_followed_at_once_while_young_then_slowly(self):
        rng = np.random.default_rng(5)
        clock = sa.ClockSync()
        local = lambda T: 77.0 + T * (1 + 300e-6)       # noqa: E731 - a Mac 300 ppm fast
        pings(clock, 25, local, rng)
        self.assertAlmostEqual(clock.server_rate(), 1 / (1 + 300e-6), delta=30e-6)     # taken at once
        pings(clock, 15, local, rng, start=25.0)
        t_a, off, before = clock.estimate
        clock.raw = (t_a, off, before + 400e-6)                           # the line fit runs off
        clock._apply(t_a + 1.0)
        self.assertAlmostEqual(clock.estimate[2] - before, 10e-6, delta=1e-9)       # 10 ppm in that second

    def test_a_far_estimate_that_comes_back_is_no_jump(self):
        clock = sa.ClockSync()
        for i in range(20):                                     # settled at offset 0
            clock.add(i, i + 0.001, i + 0.001, i + 0.002)
        t_a, off, _ = clock.estimate
        clock.raw = (19.5, 0.025, 0.0)                          # two estimates far off...
        clock._apply(20.0)
        clock._apply(20.2)
        clock.raw = (20.4, 0.0, 0.0)                            # ...then back
        clock._apply(20.4)
        self.assertEqual(clock.steps, 0)
        self.assertLess(abs(clock.estimate[1]), 200e-6 * 1.5)

    def test_the_first_estimate_is_taken_as_it_is(self):
        clock = sa.ClockSync()
        for i in range(sa.ClockSync.MIN_SAMPLES):
            clock.add(i, 0.3 + i + 0.001, 0.3 + i + 0.001, i + 0.002)     # 0.3 s apart
        self.assertAlmostEqual(clock.estimate[1], 0.3, places=4)
        self.assertEqual(clock.steps, 0)

    def test_ping_log_and_stats(self):
        clock, local, rng = self.settled()
        pings(clock, 60, local, rng, start=40.0)
        self.assertEqual(len(clock.ping_log), 200)
        t4, off_ms, rtt_ms, used = clock.ping_log[-1]
        self.assertGreater(rtt_ms, 2.0)
        self.assertLess(abs(off_ms + 500_000), 10)                         # the test's clocks are 500 s apart
        self.assertGreater(sum(u for *_, u in clock.ping_log), 20)          # the fastest 30 % are used
        self.assertLess(sum(u for *_, u in clock.ping_log), 120)
        st = clock.stats()
        self.assertEqual(set(st), {"clock_offset_ms", "clock_drift_ppm", "rtt_ms", "clock_slew_ms_s", "clock_steps"})
        self.assertLess(abs(st["clock_offset_ms"] + 500_000), 1.0)


if __name__ == "__main__":
    unittest.main()
