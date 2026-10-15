"""Capture (BlackHole -> CaptureEngine) in virtual time: every block's index is time, whatever macOS
or Python does. The virtual input device captures its own frame index (vdevice.code), so each
captured sample says which device frame it was."""
import ctypes
import os
import sys
import unittest

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
sys.path.insert(0, here)
import audio_engine as ae  # noqa: E402
import sacore  # noqa: E402
import vdevice as vd  # noqa: E402

RATE = vd.RATE
LEAD = 0.005                # ADC time = the IO cycle's time minus this


class Capture:
    """AudioInput on a virtual BlackHole, its engine ticking every 5 ms except during `stalls`."""

    def __init__(self, sim, sd, after=None, stalls=()):
        self.blocks = []
        self.input = ae.AudioInput(sd, 0, RATE, 2, lambda pcm, n, t: self.blocks.append((n, t, pcm.copy())),
                                   after=after, now=lambda: sim.t, threads=False)
        self.device = self.input.stream.device
        sim.every(ae.CaptureEngine.PERIOD, self.input.capture.tick,
                  skip=lambda t: not self.input.stream.active or any(a <= t < a + d for a, d in stalls))

    def frames(self):
        """(device frame each captured sample says it is, its n, its stamp)."""
        n = np.concatenate([np.arange(b[0], b[0] + len(b[2])) for b in self.blocks])
        code = vd.decode(np.concatenate([b[2][:, 0] for b in self.blocks]))
        stamps = np.array([(b[0], b[1]) for b in self.blocks])
        return n, code, stamps


def blackhole(**kw):
    return [{"name": "BlackHole 2ch", "max_input_channels": 2, "lead": LEAD, **kw}]


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.sim = vd.Sim()
        self.sd = vd.VirtualSD(self.sim, blackhole())

    def test_skipped_input_cycles_are_zeros_in_their_place(self):
        c = Capture(self.sim, self.sd)
        self.sim.every(2.0, lambda: c.device.drop(512), first=self.sim.t + 1.0)
        self.sim.run(9)
        n, code, stamps = c.frames()
        self.assertTrue(np.array_equal(np.diff(n), np.ones(len(n) - 1)))        # n never jumps
        skipped = np.zeros(len(n), bool)
        f = n - n[0]                                    # the device frame of each sample (n starts at 0)
        for t, k in c.device.dropped:
            f0 = round((t - c.device.io0) * RATE)       # the dropped frames start where the IO cycle was due
            skipped |= (f >= f0) & (f < f0 + k)
        self.assertEqual(len(c.device.dropped), 4)
        self.assertEqual(int(skipped.sum()), 4 * 512)
        self.assertTrue(np.all(code[skipped] == vd.decode(0.0)))      # zeros
        self.assertTrue(np.array_equal(code[~skipped], (f[~skipped] % vd.CODE)))
        self.assertEqual(c.input.second()["cap_skips"], 4)
        # stamps on one line: sample n was captured at io0 - LEAD + n / RATE
        line = c.device.io0 - LEAD + stamps[:, 0] / RATE
        self.assertLess(np.abs(stamps[:, 1] - line).max(), 1e-6)
        c.input.close()

    def test_falling_a_ring_behind_loses_audio_not_time(self):
        c = Capture(self.sim, self.sd, stalls=[(self.sim.t + 2.0, 6.0)])      # Python stuck for 6 s
        self.sim.run(12)
        n, code, stamps = c.frames()
        f = n - n[0]
        self.assertTrue(np.array_equal(code, f % vd.CODE))      # every sample is the frame its n says
        jumps = np.flatnonzero(np.diff(n) != 1)
        self.assertEqual(len(jumps), 1)                         # one stretch lost...
        lost = n[jumps[0] + 1] - n[jumps[0]] - 1
        self.assertGreater(lost, 6.0 * RATE - c.input.ring.cap)
        self.assertEqual(c.input.second()["cap_overflows"], 1)  # ...and counted
        line = c.device.io0 - LEAD + stamps[:, 0] / RATE        # time never moved
        self.assertLess(np.abs(stamps[:, 1] - line).max(), 1e-6)
        c.input.close()

    def test_a_stall_of_a_ring_inside_a_tick_loses_that_ring_once(self):
        # The thread stalls right after it looked at the ring (waiting for the GIL, say) while the device
        # captures more than a ring: the position it jumps to must be behind the newest frame NOW (one
        # behind the frame it saw is gone too, and it would retry it forever)
        ring = sacore.Ring("input", 2, RATE)
        callback, t = sacore.CALLBACK(ring.callback), [100.0]
        pcm = np.zeros((512, 2), np.float32)

        def capture(buffers):
            for _ in range(buffers):
                callback(pcm.ctypes.data, None, 512, ctypes.byref(sacore.TimeInfo(t[0], t[0] + 0.001, 0.0)), 0, ring.h)
                t[0] += 512 / RATE
        got = []
        cap = ae.CaptureEngine(ring, lambda x, n, at: got.append(n))
        for _ in range(4):
            capture(4)
            cap.tick()
        capture(2)
        state, reads = ring.state, []

        def stalled_state():                        # (only the first look stalls)
            ring.state = state
            st = state()
            capture(ring.cap // 512 + 8)
            return st
        def counted_read(pos, n, read=ring.read):
            reads.append(pos)
            if len(reads) > 1000:
                raise AssertionError(f"the tick never ends: still reading at {pos}, lapped {cap.lapped} times")
            return read(pos, n)
        ring.read, ring.state = counted_read, stalled_state
        try:
            cap.tick()
        finally:
            ring.close()
        self.assertEqual(cap.lapped, 1)
        self.assertTrue(np.all(np.diff(got) > 0))               # n still moves on with time

    def test_a_gap_in_blackhole_itself_is_told_from_a_stall_of_python(self):
        # missed(): what the session excuses (DelayPolicy.excuse). BlackHole calling back late or not
        # at all is; the capture thread stalling (Python) isn't - the C ring knows which
        c = Capture(self.sim, self.sd, stalls=[(self.sim.t + 7.0, 0.5)])
        T0, seen = self.sim.t, []
        self.sim.every(0.25, lambda: seen.append((round(self.sim.t - T0, 2), c.input.missed())), first=T0 + 1.0)
        self.sim.at(T0 + 3.0, lambda: c.device.drop(int(0.6 * RATE)))     # its IO stops for 0.6 s
        self.sim.run(9)
        missed = [t for t, m in seen if m]
        self.assertEqual(missed, [3.25, 3.5, 3.75], seen)     # late twice, then the skip it counted
        c.input.close()

    def test_a_reopened_device_carries_on_in_real_time(self):
        a = Capture(self.sim, self.sd)
        self.sim.advance(5.0)
        after = a.input.continuity()
        a.input.close()
        self.sim.advance(0.3)                       # PortAudio refresh: the device is away for 0.3 s
        b = Capture(self.sim, self.sd, after=after)
        self.sim.advance(5.0)
        stamps = np.array([(n, t) for n, t, _ in a.blocks + b.blocks])
        self.assertTrue(np.all(np.diff(stamps[:, 0]) > 0))
        line = stamps[0, 1] + (stamps[:, 0] - stamps[0, 0]) / RATE
        self.assertLess(np.abs(stamps[:, 1] - line).max(), 1.0 / RATE)     # one line across the reopen
        gap = b.input.second()["cap_gap_frames"]
        self.assertAlmostEqual(gap, 0.3 * RATE, delta=2 * 512)              # the gap is the real time away
        self.assertEqual(gap, b.blocks[0][0] - after[0])
        b.input.close()


if __name__ == "__main__":
    unittest.main()
