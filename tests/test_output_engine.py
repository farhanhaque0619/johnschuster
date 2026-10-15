"""The output engine against the real C callbacks, in virtual time (tests/vdevice.py), judged at the
speaker: what each frame played against what was due at that true moment.

T1-T13 of the 2.0 spec: steady play, clock drift, report wobble, skipped IO cycles, latency steps,
losses too small to see, Python and network stalls, delay changes on two Macs, glides, volume and
channel changes, PortAudio's own buffer sizes, and the ring's limits.
"""
import math
import os
import sys
import unittest

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
sys.path.insert(0, here)
import sacore  # noqa: E402
import audio_engine as ae  # noqa: E402
import vdevice as vd  # noqa: E402

RATE = vd.RATE


def ou(seconds, frames, sigma, tau=1.0, seed=0):
    """Report wobble that wanders (Bluetooth): an Ornstein-Uhlenbeck sequence, one value per callback."""
    rng = np.random.default_rng(seed)
    a = math.exp(-frames / RATE / tau)
    x = np.zeros(int(seconds * RATE / frames) + 100)
    for i in range(1, len(x)):
        x[i] = a * x[i - 1] + math.sqrt(1 - a * a) * sigma * rng.normal()
    return lambda k: float(x[min(k, len(x) - 1)])


class Mac:
    """One Mac playing the host's stream: its own clock (`off`, `drift_ppm` from the host's), its
    output device, the ring, the StreamBuffer and the OutputEngine, ticking every 10 ms except while
    `stalls` hold its Python up."""

    def __init__(self, sim, seconds, frames=1024, lead=0.2, ppm=0.0, off=0.0, drift_ppm=0.0, ring_s=4.0,
                 channels=2, stalls=(), report=lambda k: 0.0, invalid=lambda k: False, **engine):
        ta, k = sim.t, drift_ppm * 1e-6
        self.S = lambda t: t + off + k * (t - ta)               # this Mac's clock -> the host's
        self.local = lambda t: ta + (t - off - ta) / (1 + k)
        self.ring = sacore.Ring("output", channels, RATE, seconds=ring_s)
        self.sb = ae.StreamBuffer(RATE, 2)
        self.engine = ae.OutputEngine(self.ring, self.sb, to_server=self.S, server_rate=lambda: 1 + k,
                                      now=lambda: self.local(sim.t), **engine)
        self.dev = vd.Device(sim, self.ring, frames, lead, ppm, seconds, local=self.local, report=report,
                             invalid=invalid)
        sim.every(ae.OutputEngine.TICK, self.engine.tick, skip=lambda t: any(a <= t < a + d for a, d in stalls))

    def judge(self, due, t_from, t_to=math.inf, allow=()):
        return vd.judge(self.dev.tape, self.dev.tau(), due, t_from, t_to, allow)

    def close(self):
        self.ring.close()
        self.sb.content.close()


def play(seconds, D=0.5, net=0.005, delay=None, feed_stalls=(), **mac):
    """One Mac on a steady stream. Returns (sim, mac, T0)."""
    sim = vd.Sim()
    m = Mac(sim, seconds, **mac)
    T0 = sim.t
    vd.Feeder(sim, m.sb, T0, delay=delay or (lambda n: D), net=net, stalls=feed_stalls)
    sim.run(seconds)
    return sim, m, T0


class SteadyTest(unittest.TestCase):
    def tearDown(self):
        self.mac.close()

    def test_t1_steady_minute_is_sample_exact(self):
        sim, self.mac, T0 = play(60)
        r = self.mac.judge(vd.steady(T0, 0.5), T0 + 1.5, T0 + 59.5)
        self.assertEqual(r["glitches"], 0, r)
        self.assertLessEqual(r["err_max_ms"], 0.01, r)
        self.assertGreater(r["good_s"], 57)
        self.assertEqual(self.mac.engine.splices, 0)
        self.assertEqual(self.mac.engine.second()["underruns"], 0)

    def test_t2_device_and_listener_clocks_drift(self):
        # the device runs 300 ppm fast; the listener's clock is 3.5 s off the host's and 150 ppm fast
        sim, self.mac, T0 = play(25, ppm=300, off=3.5, drift_ppm=150)
        r = self.mac.judge(vd.steady(T0, 0.5), T0 + 10, T0 + 24.5)
        self.assertEqual(r["glitches"], 0, r)
        self.assertLessEqual(r["err_max_ms"], 0.05, r)

    def test_t3_report_wobble_never_reaches_the_music(self):
        # Bluetooth's DAC times wander +-2 ms (OU, sigma 1 ms, 1 s): the music stays put
        sim, self.mac, T0 = play(40, report=ou(40, 1024, 0.001))
        r = self.mac.judge(vd.steady(T0, 0.5), T0 + 10, T0 + 39.5)
        self.assertEqual(r["glitches"], 0, r)
        self.assertLessEqual(r["err_max_ms"], 2.5, r)
        self.assertEqual(self.mac.engine.splices, 0)
        self.assertEqual(self.mac.engine.second()["skips"], 0)

    def test_t12_any_buffer_size_portaudio_picks(self):
        # level 2 of the ladder: blocksize 0, PortAudio's own sizes (here 470 and 471 in turn)
        sim, self.mac, T0 = play(20, frames=[470, 471])
        r = self.mac.judge(vd.steady(T0, 0.5), T0 + 1.5, T0 + 19.5)
        self.assertEqual(r["glitches"], 0, r)
        self.assertLessEqual(r["err_max_ms"], 0.01, r)
        self.assertEqual(self.mac.engine.second()["io"], [470, 471])

    def test_missing_timestamps_play_on_now_plus_the_usual_lead(self):
        # PortAudio gives no DAC time (0) for whole stretches: "now" + the last good lead stands in
        sim, self.mac, T0 = play(20, invalid=lambda k: 200 <= k < 500)
        r = self.mac.judge(vd.steady(T0, 0.5), T0 + 1.5, T0 + 19.5)
        self.assertEqual(r["glitches"], 0, r)
        self.assertLessEqual(r["err_max_ms"], 0.01, r)
        self.assertGreater(self.mac.engine.clock.invalid, 250)


class DeviceTroubleTest(unittest.TestCase):
    def tearDown(self):
        self.mac.close()

    def test_t4_skipped_io_cycles_cost_their_own_audio_and_no_time(self):
        sim = vd.Sim()
        self.mac = m = Mac(sim, 22)
        T0 = sim.t
        vd.Feeder(sim, m.sb, T0)
        sim.every(2.0, lambda: m.dev.drop(1024), first=T0 + 3.01)
        sim.run(22)
        drops = [t for t, _ in m.dev.dropped]
        self.assertEqual(m.engine.second()["skips"], len(drops))
        r = m.judge(vd.steady(T0, 0.5), T0 + 1.5, T0 + 21.5)
        self.assertLessEqual(r["glitches"], 2 * len(drops), r)
        # after each skip, exact again
        for t in drops[:-1]:
            after = (r["err_t"] > t + 0.5) & (r["err_t"] < t + 1.9)
            self.assertLessEqual(np.abs(r["err"][after]).max() * 1000, 0.05, (t, r))

    def test_t5_latency_step_up_is_one_skip(self):
        sim = vd.Sim()
        self.mac = m = Mac(sim, 16)
        T0 = sim.t
        vd.Feeder(sim, m.sb, T0)
        sim.at(T0 + 8.0, lambda: m.dev.latency_step(0.040))
        sim.run(16)
        st = m.engine.second()
        self.assertEqual(st["skips"], 1, st)
        self.assertAlmostEqual(st["skipped_ms"], 40.0, delta=0.1)
        r = m.judge(vd.steady(T0, 0.5), T0 + 8.3, T0 + 15.5)
        self.assertLessEqual(r["err_max_ms"], 0.05, r)
        self.assertEqual(m.engine.splices, 0)

    def test_t5_at_airpods_buffers_a_latency_step_is_not_a_whole_buffer(self):
        # 40 ms is 0.94 of a 2048-frame buffer: still skipped as measured (the IO cycles came on time)
        sim = vd.Sim()
        self.mac = m = Mac(sim, 20, frames=2048)
        T0 = sim.t
        vd.Feeder(sim, m.sb, T0)
        sim.at(T0 + 8.0, lambda: m.dev.latency_step(0.040))
        sim.run(20)
        st = m.engine.second()
        self.assertEqual(st["skips"], 1, st)
        self.assertAlmostEqual(st["skipped_ms"], 40.0, delta=0.1)
        r = m.judge(vd.steady(T0, 0.5), T0 + 8.3, T0 + 19.5)
        self.assertLessEqual(r["err_max_ms"], 0.05, r)

    def test_t5b_latency_step_down_is_one_reanchor_and_one_splice(self):
        sim = vd.Sim()
        self.mac = m = Mac(sim, 16)
        T0 = sim.t
        vd.Feeder(sim, m.sb, T0)
        sim.at(T0 + 8.0, lambda: m.dev.latency_step(-0.040))
        sim.run(16)
        self.assertEqual(m.engine.clock.reanchors, 1)
        self.assertEqual(m.engine.splices, 1)
        r = m.judge(vd.steady(T0, 0.5), T0 + 9.0, T0 + 15.5)
        self.assertLessEqual(r["err_max_ms"], 0.05, r)
        self.assertEqual(r["glitches"], 0, r)

    def losses(self, seconds, frames, size, rate):
        """The HAL drops `size`-frame slices under our buffer, `rate` of the time: each one under the C
        side's half-buffer threshold. Returns the engine's last second and the judge from 10 s."""
        sim = vd.Sim()
        self.mac = m = Mac(sim, seconds, frames=frames)
        T0 = sim.t
        vd.Feeder(sim, m.sb, T0)
        rng = np.random.default_rng(1)

        def lose():
            m.dev.drop(size)
            sim.at(sim.t + size / rate / RATE * rng.uniform(0.7, 1.3), lose)
        sim.at(T0 + 0.5, lose)
        sim.run(seconds)
        return m.engine.second(), m.judge(vd.steady(T0, 0.5), T0 + 10, T0 + seconds - 0.5)

    def test_t6_losses_too_small_to_see_are_flagged_lossy(self):
        # macOS 27 Bluetooth: 256-frame slices under our 1024-frame buffer, 0.7 % of the time
        st, r = self.losses(30, 1024, 256, 0.007)
        self.assertTrue(st["lossy"], st)
        self.assertAlmostEqual(st["loss_pct"], 0.7, delta=0.2)
        self.assertEqual(st["skips"], 0)
        self.assertLessEqual(r["err_p99_ms"], 10, r)

    def test_t6_at_airpods_buffers_and_past_the_rate_clamp(self):
        # 512-frame slices under AirPods' 2048 (each over REANCHOR: a re-anchor), and 2 % at 1024 (more
        # than the narrow loop's 1 % clamp): lossy too, not a re-anchor and a splice every second or two
        for frames, size, rate in ((2048, 512, 0.007), (1024, 256, 0.02)):
            st, r = self.losses(40, frames, size, rate)
            self.assertTrue(st["lossy"], (frames, st))
            self.assertLessEqual(st["splices"], 3, (frames, st))
            self.assertLessEqual(r["err_p99_ms"], 10, (frames, r))
            self.mac.close()


class StallTest(unittest.TestCase):
    def tearDown(self):
        self.mac.close()

    def test_t7_python_stall_shorter_than_ahead_is_inaudible(self):
        sim, self.mac, T0 = play(14, stalls=[(1000.0 + 6.0, 0.2)])
        st = self.mac.engine.second()
        self.assertEqual(st["underruns"], 0, st)
        r = self.mac.judge(vd.steady(T0, 0.5), T0 + 1.5, T0 + 13.5)
        self.assertEqual(r["glitches"], 0, r)
        self.assertGreater(self.mac.engine.ahead, 0.3)          # and it keeps more in the ring now

    def test_t7b_longer_stall_is_one_underrun_then_exact(self):
        sim, self.mac, T0 = play(14, stalls=[(1000.0 + 6.0, 0.45)])
        e = self.mac.engine
        st = e.second()
        self.assertEqual(st["underruns"], 1, st)
        self.assertEqual(e.reseeks, 1)
        self.assertGreaterEqual(e.ahead, 0.45)                  # 1.5 x after the underrun, or more
        r = self.mac.judge(vd.steady(T0, 0.5), T0 + 1.5, T0 + 13.5)
        self.assertEqual(r["glitches"], 1, r)
        r = self.mac.judge(vd.steady(T0, 0.5), T0 + 6.45 + 0.5, T0 + 13.5)
        self.assertLessEqual(r["err_max_ms"], 0.05, r)
        self.assertEqual(r["glitches"], 0, r)

    def test_a_stall_of_this_macs_python_is_counted_once(self):
        # One GIL: the network thread stalls with the engine (and catches up a tick later), so what has
        # arrived lags by the stall when the engine wakes. That dip is the stall itself, which spare
        # already allows 1.5 x for.
        sim, self.mac, T0 = play(14, stalls=[(1000.0 + 6.0, 0.2)], feed_stalls=[(1000.0 + 6.0, 0.21)])
        e = self.mac.engine
        allowance = e.stall_allowance()
        self.assertGreater(allowance, 1.5 * 0.2)
        self.assertGreater(e.headroom, 0.25)                                    # 0.5 - 0.2 to the ear - net
        self.assertGreater(min(e._spare30), e.headroom - 0.03 - allowance)      # not 0.2 s less again
        r = self.mac.judge(vd.steady(T0, 0.5), T0 + 1.5, T0 + 13.5)
        self.assertEqual(r["glitches"], 0, r)

    def test_a_stall_that_leaves_less_than_a_buffer_is_a_counted_dropout(self):
        # AirPods at their 0.59 s start delay: a 0.28 s stall leaves some audio in the ring, but less than
        # the device takes next. The re-seek's new run starts past it, and the silence between (~70 ms)
        # is a dropout like any other: counted, and `ahead` grows
        sim, self.mac, T0 = play(10, D=0.59, frames=2048, stalls=[(1000.0 + 6.0, 0.28)])
        e = self.mac.engine
        st = e.second()
        r = self.mac.judge(vd.steady(T0, 0.59), T0 + 2, T0 + 9.5)
        self.assertEqual(r["glitches"], 1, r)
        self.assertGreater(r["holes_ms"], 60, r)
        self.assertEqual((e.reseeks, st["underruns"]), (1, 1), st)
        self.assertGreater(st["underrun_ms"], 60, st)
        self.assertGreater(e.ahead, 0.4)

    def test_4096_frame_buffers_keep_a_buffer_and_the_guard_in_the_ring(self):
        # AirPods at IO level 1. With `ahead` at its 0.15 s floor, a 4096-frame callback left less than
        # the guard behind it, and every callback re-seeked (half the audio lost, nothing counted)
        sim = vd.Sim()
        self.mac = m = Mac(sim, 20, frames=4096, lead=0.25)
        m.engine.ahead = ae.OutputEngine.AHEAD_MIN              # where 60 quiet seconds shrink it to
        T0 = sim.t
        vd.Feeder(sim, m.sb, T0, delay=lambda n: 0.75)
        sim.run(20)
        r = m.judge(vd.steady(T0, 0.75), T0 + 2, T0 + 19.5)
        self.assertEqual((r["glitches"], m.engine.reseeks), (0, 0), r)
        self.assertGreaterEqual(m.engine.ahead, (2 * 4096 + 64) / RATE)

    def test_t8_network_stall_inside_the_headroom_is_inaudible(self):
        # D 0.57 s - 0.2 s from the ring to the ear - 15 ms capture and network = 355 ms of headroom
        # (its minimum is a 21 ms buffer less: just after the device takes one)
        sim, self.mac, T0 = play(14, D=0.57, feed_stalls=[(1000.0 + 6.0, 0.2)])
        self.assertAlmostEqual(self.mac.engine.second()["headroom_min_ms"], 335, delta=20)  # less a buffer
        r = self.mac.judge(vd.steady(T0, 0.57), T0 + 1.5, T0 + 13.5)
        self.assertEqual(r["glitches"], 0, r)

    def test_t8b_network_stall_past_the_headroom_is_one_gap_then_exact(self):
        sim, self.mac, T0 = play(14, D=0.57, feed_stalls=[(1000.0 + 6.0, 0.45)])
        r = self.mac.judge(vd.steady(T0, 0.57), T0 + 1.5, T0 + 13.5)
        self.assertLessEqual(r["glitches"], 1, r)
        self.assertAlmostEqual(self.mac.engine.ahead, 0.45, delta=0.01)     # 1.5 x after the underrun
        r = self.mac.judge(vd.steady(T0, 0.57), T0 + 7.2, T0 + 13.5)
        self.assertEqual(r["glitches"], 0, r)
        self.assertLessEqual(r["err_max_ms"], 0.05, r)


class DelayChangeTest(unittest.TestCase):
    """The host changes the delay from sample n_s on (DelaySchedule stamps it); every Mac switches at
    that sample, at the same host time."""

    N_S = 8 * RATE                  # the change lands on the block at 8 s of content

    def setUp(self):
        self.macs = []

    def tearDown(self):
        for m in self.macs:
            m.close()

    def two_macs(self, seconds, delay, D=0.5):
        sim = vd.Sim()
        host = Mac(sim, seconds, frames=512, ppm=-40)                                      # its speakers
        friend = Mac(sim, seconds, frames=2048, ppm=120, off=3.5, drift_ppm=150, lead=0.16)  # AirPods
        self.macs = [host, friend]
        T0 = sim.t
        vd.Feeder(sim, host.sb, T0, delay=delay, net=0.0005)
        vd.Feeder(sim, friend.sb, T0, delay=delay, net=0.012)
        sim.run(seconds)
        return T0, host, friend

    def test_t9_step_up_is_the_same_silence_at_the_same_moment_on_both(self):
        T0, *macs = self.two_macs(14, lambda n: 0.5 if n < self.N_S else 0.62)
        old_end = T0 + self.N_S / RATE + 0.5            # where content n_s would have played
        new = old_end + 0.12

        def due(s):
            gap = (s >= old_end) & (s < new)
            return np.where(s < old_end, (s - T0 - 0.5) * RATE, (s - T0 - 0.62) * RATE), gap
        for m in macs:
            fades = [(old_end - 0.006, old_end), (new, new + 0.006)]
            r = m.judge(due, T0 + 1.5, T0 + 13.5, allow=fades)
            self.assertEqual(r["glitches"], 0, r)
            r = m.judge(due, T0 + 6, T0 + 13.5, allow=fades)     # once the speed loop has settled
            self.assertLessEqual(r["err_max_ms"], 0.05, r)
            self.assertEqual((m.engine.steps, m.engine.splices), (1, 0))
            # the silence itself: frames at exactly zero from where content n_s would have played on
            # the old line (the fade-out's last frame, at gain 0) to where it plays on the new line
            # (the fade-in's first): 0.120 s, to a frame of the device's own grid
            tau = m.dev.tau()
            i = np.nonzero((tau > old_end - 0.01) & (tau < new + 0.01))[0]
            zero = np.abs(m.dev.tape[i]).max(1) == 0
            edges = np.flatnonzero(np.diff(zero.astype(int)))
            first, last = i[edges[0] + 1], i[edges[1]]
            frame = 1 / m.dev.true_rate
            self.assertAlmostEqual(tau[last] - tau[first], 0.12, delta=1.5 * frame)
            self.assertAlmostEqual(tau[first], old_end, delta=1.5 * frame)
            self.assertAlmostEqual(tau[last], new, delta=1.5 * frame)

    def test_an_increase_fades_out_and_in_over_5ms_whatever_the_timing(self):
        """Even when the producer is right behind the network (headroom under `ahead`), and wherever
        the due frames fall in its blocks: a full 240-frame fade on both sides of the added silence."""
        for k in range(8):
            D = 0.36 + k * 0.00037                  # 0.2 s of it to the ear: 0.15 s of headroom, then 0.27
            sim = vd.Sim()
            m = Mac(sim, 11, frames=512)
            self.macs = [m]
            T0 = sim.t
            vd.Feeder(sim, m.sb, T0, delay=lambda n, D=D: D if n < self.N_S else D + 0.12)
            sim.run(11)
            self.assertLess(m.engine.headroom, m.engine.ahead)
            old_end = T0 + self.N_S / RATE + D
            tau = m.dev.tau()

            def gain(t, line):                      # how loud the content due at t plays (0..1)
                i = int(np.argmin(np.abs(tau - t)))
                want = vd.signal(int(round((tau[i] - T0 - line) * RATE)), 1)[0].astype(np.float64)
                return float(np.dot(m.dev.tape[i], want) / np.dot(want, want))
            frame = 1 / RATE
            for t, line, g in ((old_end - 250 * frame, D, 1.0), (old_end - 120 * frame, D, 0.5),
                               (old_end + 0.12 + 120 * frame, D + 0.12, 0.5), (old_end + 0.12 + 250 * frame, D + 0.12, 1.0)):
                self.assertAlmostEqual(gain(t, line), g, delta=0.03, msg=(k, round(t - old_end, 5)))
            m.close()
        self.macs = []

    def test_t9b_step_down_inside_silence_is_one_quiet_switch(self):
        quiet = (int(7.5 * RATE), self.N_S + int(0.3 * RATE))     # the host only lowers inside silence

        def content(n, k):
            pcm = vd.signal(n, k)
            pcm[(np.arange(n, n + k) >= quiet[0]) & (np.arange(n, n + k) < quiet[1])] = 0
            return pcm
        sim = vd.Sim()
        m = Mac(sim, 14, frames=1024)
        self.macs = [m]
        T0 = sim.t
        vd.Feeder(sim, m.sb, T0, delay=lambda n: 0.6 if n < self.N_S else 0.52, content=content)
        sim.run(14)
        due_s = T0 + self.N_S / RATE + 0.52

        def due(s):
            gap = (s >= T0 + quiet[0] / RATE + 0.6) & (s < T0 + quiet[1] / RATE + 0.52)   # silent content
            return np.where(s < due_s, (s - T0 - 0.6) * RATE, (s - T0 - 0.52) * RATE), gap
        r = m.judge(due, T0 + 1.5, T0 + 13.5)
        self.assertEqual(r["glitches"], 0, r)
        self.assertLessEqual(r["err_max_ms"], 0.05, r)
        self.assertEqual((m.engine.steps, m.engine.splices), (1, 0))

    def test_t9b_step_down_crossfades_exactly_when_the_new_line_is_due(self):
        sim = vd.Sim()
        m = Mac(sim, 14, frames=1024)
        self.macs = [m]
        T0 = sim.t
        vd.Feeder(sim, m.sb, T0, delay=lambda n: 0.6 if n < self.N_S else 0.52)
        sim.run(14)
        due_s = T0 + self.N_S / RATE + 0.52

        def due(s):
            return np.where(s < due_s, (s - T0 - 0.6) * RATE, (s - T0 - 0.52) * RATE), np.zeros(len(s), bool)
        # the old line plays to the frame before due_s, the 240-frame crossfade starts on due_s
        r = m.judge(due, T0 + 1.5, T0 + 13.5, allow=[(due_s, due_s + 241 / RATE)])
        self.assertEqual(r["glitches"], 0, r)
        self.assertLessEqual(r["err_max_ms"], 0.05, r)
        self.assertEqual((m.engine.steps, m.engine.splices), (1, 0))

    def decrease(self, D0, D1, frames, quiet=None):
        """One Mac, the delay going down from D0 to D1 at N_S (inside silent content `quiet`, if given):
        judged against both lines, the crossfade allowed where the new one is due."""
        def content(n, k):
            pcm = vd.signal(n, k)
            if quiet:
                pcm[(np.arange(n, n + k) >= quiet[0]) & (np.arange(n, n + k) < quiet[1])] = 0
            return pcm
        sim = vd.Sim()
        m = Mac(sim, 16, frames=frames)
        self.macs = [m]
        T0 = sim.t
        vd.Feeder(sim, m.sb, T0, delay=lambda n: D0 if n < self.N_S else D1, content=content)
        sim.run(16)
        due_s = T0 + self.N_S / RATE + D1

        def due(s):
            gap = np.zeros(len(s), bool)
            if quiet:
                gap = (s >= T0 + quiet[0] / RATE + D0) & (s < T0 + quiet[1] / RATE + D1)
            return np.where(s < due_s, (s - T0 - D0) * RATE, (s - T0 - D1) * RATE), gap
        return m, m.judge(due, T0 + 1.5, T0 + 15.5, allow=[(due_s, due_s + 241 / RATE)])

    def test_t9b_step_down_with_less_headroom_after_it_than_ahead(self):
        # 0.6 -> 0.5 s: afterwards the audio that has arrived reaches only ~0.28 s past the device, less
        # than the 0.3 s the ring held of the old line; that part of the ring must not play
        m, r = self.decrease(0.6, 0.5, 1024)
        self.assertEqual(r["glitches"], 0, r)
        self.assertLessEqual(r["err_max_ms"], 0.05, r)
        self.assertEqual((m.engine.steps, m.engine.splices), (1, 0))

    def test_t9b_a_lower_in_silence_on_airpods_plays_the_next_song_exactly(self):
        # what DelayPolicy does: 100 ms lower, inside 0.45 s of silence, here at 2048 frames
        m, r = self.decrease(0.59, 0.49, 2048, quiet=(self.N_S - int(0.45 * RATE), self.N_S + 480))
        self.assertEqual(r["glitches"], 0, r)
        self.assertLessEqual(r["err_max_ms"], 0.05, r)
        self.assertEqual((m.engine.steps, m.engine.splices), (1, 0))

    def test_a_change_right_after_the_entry_point_is_played_as_stamped(self):
        """Starting to play (or re-entering) just before a delay change: the entry block stops for it
        too, wherever the entry point falls in the device's buffers."""
        for k in range(10):
            D0 = 0.5 + k * 0.0021
            sim = vd.Sim()
            m = Mac(sim, 3, frames=1024)
            self.macs = [m]
            T0 = sim.t
            vd.Feeder(sim, m.sb, T0, delay=lambda n, D0=D0: D0 if n < 480 else D0 + 0.1)
            sim.run(3)
            old_end, new = T0 + 480 / RATE + D0, T0 + 480 / RATE + D0 + 0.1

            def due(s, D0=D0, old_end=old_end, new=new):
                gap = (s >= old_end) & (s < new)
                return np.where(s < old_end, (s - T0 - D0) * RATE, (s - T0 - D0 - 0.1) * RATE), gap
            r = m.judge(due, T0 + D0 + 0.007, T0 + 2.5, allow=[(old_end - 0.006, old_end), (new, new + 0.006)])
            self.assertEqual((r["glitches"], m.engine.steps, m.engine.splices), (0, 1, 0), (k, r))
            m.close()
        self.macs = []

    def test_t10_glide_follows_the_stamps_without_a_splice(self):
        g, start = 0.0025, 4 * RATE         # the delay grows 25 us per 10 ms block, up to +100 ms
        glide = lambda n: min(0.1, g * max(0, n - start) / RATE)     # noqa: E731
        sim = vd.Sim()
        m = Mac(sim, 50, frames=2048, lead=0.16)
        self.macs = [m]
        T0 = sim.t
        f = vd.Feeder(sim, m.sb, T0, delay=lambda n: 0.5 + glide(n // vd.BLOCK * vd.BLOCK))
        sim.run(50)
        an = np.array(f.stamps)

        def due(s):                         # invert the stamps (straight between blocks)
            return np.interp(s, an[:, 1], an[:, 0]), np.zeros(len(s), bool)
        r = m.judge(due, T0 + 1.5, T0 + 49.5)
        self.assertEqual(m.engine.splices, 0)
        self.assertEqual(r["glitches"], 0, r)
        self.assertLessEqual(r["err_max_ms"], 0.5, r)
        # the speed never strays more than the glide needs (0.25 %)
        self.assertLessEqual(1 - r["speed"][0], 0.0026, r)
        self.assertLessEqual(r["speed"][1] - 1, 0.0026, r)


class MixTest(unittest.TestCase):
    """Volume and channel changes glide over a block; a mono device gets the mix."""

    def tearDown(self):
        self.mac.close()

    def run_dc(self, channels=2, volume=lambda t: 1.0, channel=lambda t: "both", seconds=6):
        """Plays a constant (0.5, -0.25); volume / channel are functions of the seconds since the start."""
        dc = lambda n, k: np.tile(np.array([[0.5, -0.25]], np.float32), (k, 1))     # noqa: E731
        sim = vd.Sim()
        since = lambda: self.mac.engine.now() - sim.start                           # noqa: E731
        self.mac = m = Mac(sim, seconds, channels=channels, volume=lambda: volume(since()),
                           channel=lambda: channel(since()))
        vd.Feeder(sim, m.sb, sim.t, content=dc)
        sim.run(seconds)
        tau = m.dev.tau()
        return m.dev.tape[(tau > sim.start + 1.0) & (tau < sim.start + seconds - 0.5)].astype(np.float64)

    def test_t11_volume_changes_glide(self):
        y = self.run_dc(volume=lambda t: 1.0 if t < 3.0 else 0.2)
        self.assertTrue(np.allclose(y[0], [0.5, -0.25]))
        self.assertTrue(np.allclose(y[-1], [0.1, -0.05]))
        self.assertLessEqual(np.abs(np.diff(y[:, 0])).max(), 0.5 / ae.OutputEngine.BLOCK + 1e-6)

    def test_t11_channel_changes_glide_and_mono_is_the_mean(self):
        y = self.run_dc(channel=lambda t: "both" if t < 3.0 else "right")
        self.assertTrue(np.allclose(y[0], [0.5, -0.25]))
        self.assertTrue(np.allclose(y[-1], [-0.25, -0.25]))
        self.assertLessEqual(np.abs(np.diff(y, axis=0)).max(), 0.75 / ae.OutputEngine.BLOCK + 1e-6)
        self.mac.close()
        y = self.run_dc(channels=1)
        self.assertTrue(np.allclose(y[:, 0], 0.125))


class RingLimitTest(unittest.TestCase):
    def test_t13_never_writes_within_4096_of_a_full_ring(self):
        sim = vd.Sim()
        m = Mac(sim, 10, ring_s=0.6)            # a 32768-frame ring, smaller than `ahead` can grow
        writes = []
        write = m.ring.write

        def checked(pos, pcm, guard):
            r = m.ring.state()["r"]
            writes.append(pos + len(pcm) - r)
            return write(pos, pcm, guard)
        m.ring.write = checked
        m.engine.ahead = m.engine.AHEAD_MIN = 1.0
        vd.Feeder(sim, m.sb, sim.t, delay=lambda n: 1.5)
        sim.run(10)
        self.assertLessEqual(max(writes), m.ring.cap - 4096)
        self.assertGreater(max(writes), m.ring.cap // 2)            # it did fill up to the limit
        r = m.judge(vd.steady(sim.start, 1.5), sim.start + 2.5, sim.start + 9.5)
        self.assertEqual(r["glitches"], 0, r)
        m.close()


if __name__ == "__main__":
    unittest.main()
