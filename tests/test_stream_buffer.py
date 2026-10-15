"""StreamBuffer: when each received sample is due, from the host's per-block stamps."""
import os
import sys
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import audio_engine as ae  # noqa: E402

RATE, B = 48000, 480
BLOCK = np.zeros((B, 2), np.float32)


def fed(stamps):
    """A StreamBuffer fed blocks 0, B, 2B, ... stamped stamps(n)."""
    sb = ae.StreamBuffer(RATE, 2)
    for k in range(200):
        sb.feed(k * B, stamps(k * B), BLOCK)
    return sb


class StreamBufferTest(unittest.TestCase):
    def test_a_step_over_1ms_starts_a_segment_at_that_sample(self):
        sb = fed(lambda n: 100 + n / RATE + (0.12 if n >= 50 * B else 0) + (0.0009 if n >= 80 * B else 0))
        self.assertEqual(sb.seg, [0, 50 * B])           # 0.9 ms is the same line, slewed
        self.assertAlmostEqual(sb.play_at(50 * B - 1), 100 + (50 * B - 1) / RATE, places=9)
        self.assertAlmostEqual(sb.play_at(50 * B), 100 + 50 * B / RATE + 0.12, places=9)
        self.assertEqual(sb.next_start(10 * B), (50 * B, sb.play_at(50 * B)))
        self.assertIsNone(sb.next_start(50 * B))

    def test_glide_stamps_are_followed_exactly(self):
        g = 0.0025                  # 25 us more delay per 10 ms block, from block 40 to block 120
        d = lambda n: 0.5 + g * min(max(n - 40 * B, 0), 80 * B) / RATE      # noqa: E731
        sb = fed(lambda n: 100 + n / RATE + d(n))
        self.assertEqual(sb.seg, [0])                   # a glide is not a delay step
        for x in np.linspace(0, 199 * B, 997):
            k = int(x // B) * B                         # straight between the block stamps
            want = 100 + k / RATE + d(k) + (x - k) * ((B / RATE + d(k + B) - d(k)) / B)
            self.assertAlmostEqual(sb.play_at(x), want, delta=1e-6)
        self.assertAlmostEqual(sb.slope(60 * B + 7), (1 + g) / RATE, delta=1e-12)
        self.assertAlmostEqual(sb.slope(10 * B), 1 / RATE, delta=1e-12)
        self.assertAlmostEqual(sb.slope(199 * B + 100), 1 / RATE, delta=1e-12)     # past the newest

    def test_index_at_inverts_play_at_and_knows_the_added_silence(self):
        sb = fed(lambda n: 100 + n / RATE + (0.12 if n >= 50 * B else 0))
        t_end_old = 100 + 50 * B / RATE                 # where n_s would have played
        for t in (100.2, t_end_old - 0.001, t_end_old + 0.2, 101.5):
            n, gap = sb.index_at(t)
            self.assertFalse(gap)
            self.assertAlmostEqual(sb.play_at(n), t, places=9)
        self.assertEqual(sb.index_at(t_end_old + 0.05), (50 * B, True))
        self.assertEqual(sb.index_at(99.0), (0, True))                  # before anything is due

    def test_a_decrease_overlaps_and_the_new_line_wins(self):
        sb = fed(lambda n: 100 + n / RATE - (0.08 if n >= 50 * B else 0))
        t_new = 100 + 50 * B / RATE - 0.08
        self.assertEqual(sb.index_at(t_new - 0.001)[1], False)
        self.assertLess(sb.index_at(t_new - 0.001)[0], 50 * B)          # still the old line
        self.assertAlmostEqual(sb.index_at(t_new + 0.001)[0], 50 * B + 0.001 * RATE, places=6)

    def test_holes_and_new_streams(self):
        sb = ae.StreamBuffer(RATE, 2)
        sb.feed(0, 100.0, BLOCK)
        sb.feed(2 * B, 100.0 + 2 * B / RATE, BLOCK)     # a block never came
        self.assertEqual(sb.holes, B)
        self.assertEqual(sb.content.range(), (0, 3 * B))
        sb.feed(0, 200.0, BLOCK)                        # the host started again from 0
        lo, end = sb.content.range()
        self.assertEqual(end - lo, B)                   # nothing of the old stream is playable
        self.assertEqual(sb.seg, [lo])
        self.assertAlmostEqual(sb.play_at(lo), 200.0)
        sb.feed(10**8, 300.0, BLOCK)                    # far ahead: also a new stream
        self.assertEqual(len(sb.seg), 1)
        self.assertAlmostEqual(sb.play_at(sb.seg[0]), 300.0)

    def test_old_anchors_are_dropped_but_the_line_stays(self):
        sb = ae.StreamBuffer(RATE, 2)
        for k in range(3000):           # 30 s of blocks, a step at 5 s and 20 s
            n = k * B
            sb.feed(n, 100 + n / RATE + 0.1 * (n >= 5 * RATE) + 0.1 * (n >= 20 * RATE), BLOCK)
        self.assertLess(len(sb.an), 1.2 * sb.KEEP * RATE / B)
        self.assertEqual(sb.seg[1:], [20 * RATE])
        self.assertEqual(sb.seg[0], sb.an[0])
        self.assertAlmostEqual(sb.play_at(25 * RATE), 125.2, places=9)
        self.assertAlmostEqual(sb.play_at(sb.an[0] + 5), 100.1 + (sb.an[0] + 5) / RATE, places=9)

    def test_no_sample_is_visible_before_its_stamp(self):
        """The output engine reads `end` and then play_at(end - 1): the stamp must already be there."""
        sb = ae.StreamBuffer(RATE, 2)
        stop, seen, bad = threading.Event(), [0], []

        def reader():
            while not stop.is_set():
                end = sb.end()
                if end is not None:
                    with sb.lock:
                        newest = sb.an[-1] if sb.an else None
                    seen[0] += 1
                    if newest is None or end > newest + B:
                        bad.append((end, newest))
        old = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        t = threading.Thread(target=reader)
        t.start()
        try:
            for k in range(20000):
                sb.feed(k * B, 100 + k * B / RATE, BLOCK)
        finally:
            stop.set()
            t.join()
            sys.setswitchinterval(old)
        self.assertGreater(seen[0], 1000)
        self.assertEqual(bad[:5], [])


if __name__ == "__main__":
    unittest.main()
