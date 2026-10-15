"""The host's delay: how long it should be (DelayPolicy) and where a change lands (DelaySchedule).

The policy sees each player's 1 Hz report; the schedule stamps every captured block. Both take time
from the caller, so these run in no time at all."""
import math
import os
import sys
import unittest

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
sys.path.insert(0, here)
import syncaudio as sa  # noqa: E402

RATE = 48000
BLOCK = 480
BLOCK_S = BLOCK / RATE
AIRPODS_LEAD = 0.160 + 2048 / RATE          # their fixed latency plus a 2048-frame buffer


class StartDelayTest(unittest.TestCase):
    def test_from_the_slowest_output_known(self):
        policy = sa.DelayPolicy(sa.DelaySchedule(0.45))
        self.assertEqual(policy.start_delay(AIRPODS_LEAD), 0.59)        # 1.1.6 got there in 4 audible jumps
        self.assertEqual(policy.start_delay(0.031), 0.45)               # speakers
        self.assertEqual(policy.start_delay(None), 0.45)                # not playing here
        remembered = sa.DelayPolicy(sa.DelaySchedule(0.45), leads=[AIRPODS_LEAD])
        self.assertEqual(remembered.start_delay(0.031), 0.59)           # AirPods were used here lately

    def test_never_past_d_max_and_not_for_a_month_of_an_airplay_speaker(self):
        airplay = sa.DelayPolicy(sa.DelaySchedule(0.45), leads=[1.9])  # one session on one, last week
        self.assertEqual(airplay.start_delay(AIRPODS_LEAD), 0.59)
        self.assertLess(airplay.d_min(), 0.4)
        self.assertEqual(airplay.start_delay(1.9), 2.0)                 # on it now: as long as it gets

    def test_the_floor_for_lowering(self):
        policy = sa.DelayPolicy(sa.DelaySchedule(0.59), leads=[AIRPODS_LEAD])
        self.assertAlmostEqual(policy.d_min(), AIRPODS_LEAD + 0.0257 + 0.12, places=3)     # ~0.35 s
        self.assertEqual(policy.clamp(0.1), policy.d_min())
        self.assertEqual(policy.clamp(5.0), 2.0)


def report(rs, spare_ms=200.0, lead_ms=30.0, underruns=0, holes=0, deficit_ms=0.0):
    return {"reported_s": rs, "spare_1s_ms": spare_ms, "lead_ms": lead_ms, "underruns": underruns,
            "holes": holes, "deficit_ms": deficit_ms}


class Player:
    """What a player reports: its spare is the delay less what it really needs (its output's lead, the
    network, its producer's stalls), so it moves with the delay, a second later."""

    def __init__(self, need=0.25, lead=0.030):
        self.need, self.lead, self.rs, self.underruns, self.holes, self.deficit = need, lead, 0, 0, 0, 0.0

    def report(self, d):
        self.rs += 1
        return report(self.rs, (d - self.need) * 1000, self.lead * 1000, self.underruns, self.holes,
                      self.deficit * 1000)


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.schedule = sa.DelaySchedule(0.45)
        self.policy = sa.DelayPolicy(self.schedule)
        self.t = 1000.0

    def run_for(self, seconds, players):
        """Ask once a second (players() -> {key: report}); every request is applied at once, as a step.
        Returns [(seconds since the start, Request)]."""
        out = []
        for _ in range(int(seconds)):
            self.t += 1.0
            req = self.policy.update(self.t, players())
            if req:
                out.append((self.t - 1000.0, req))
                self.apply(req)
        return out

    def apply(self, req):
        self.schedule.d += req.delta
        self.schedule.changed_at = self.t

    def reports(self, **players):
        return lambda: {k: p.report(self.schedule.d) for k, p in players.items()}

    def test_a_newcomer_on_airpods_gets_room_once_before_it_plays(self):
        here = Player()
        self.assertEqual(self.run_for(10, self.reports(here=here)), [])
        # HELLO says AirPods (~0.2 s to the ear): asked on the very next update, before its first STATS
        reqs = self.run_for(1, lambda: {"here": here.report(self.schedule.d), 7: {"lead_ms": AIRPODS_LEAD * 1000}})
        self.assertEqual([(r.mode, round(r.delta, 3)) for _, r in reqs], [("join", 0.14)])
        # then its reports keep coming: nothing more (no second join, and nothing to raise)
        airpods = Player(need=AIRPODS_LEAD + 0.25, lead=AIRPODS_LEAD)
        self.assertEqual(self.run_for(60, self.reports(here=here, airpods=airpods)), [])
        speakers = self.run_for(1, lambda: {"here": here.report(self.schedule.d), 8: {"lead_ms": 35.0}})
        self.assertEqual(speakers, [])                                          # speakers need nothing

    def test_running_dry_is_urgent_once_per_5_s(self):
        p = Player()
        reqs = []
        for i in range(12):
            p.underruns += 1                                                    # every second
            reqs += self.run_for(1, self.reports(a=p))
        # (the first report is what later ones are compared with)
        self.assertEqual([(t, r.mode) for t, r in reqs], [(2.0, "urgent"), (7.0, "urgent"), (12.0, "urgent")])
        self.assertTrue(all(round(r.delta, 3) == 0.02 for _, r in reqs))       # the least: it had spare

    def test_a_close_call_is_raised_quietly_not_urgently(self):
        # 10 ms spare, nothing lost: an audible step on every Mac would be worse than the risk
        reqs = self.run_for(30, self.reports(a=Player(need=0.44)))
        self.assertEqual([(t, r.mode, round(r.delta, 3)) for t, r in reqs], [(10.0, "raise", 0.05)])

    def test_urgent_by_how_much_is_missing(self):
        p = Player(need=0.25)
        self.run_for(1, self.reports(a=p))
        p.need, p.deficit, p.underruns = 1.2, 0.9, 1                           # a 0.9 s Wi-Fi stall: ran dry
        reqs = self.run_for(6, self.reports(a=p))
        self.assertEqual([(t, r.mode, round(r.delta, 3)) for t, r in reqs], [(2.0, "urgent", 0.3)])  # (one stall
        p.need, p.deficit = 0.25, 0.0                                                   # isn't the new normal)
        self.t += 10
        p.holes += 480                                                          # audio that never arrived
        self.assertEqual([(r.mode, round(r.delta, 3)) for _, r in self.run_for(1, self.reports(a=p))],
                         [("urgent", 0.02)])                                    # the least: it has spare

    def test_a_gap_in_this_macs_own_capture_is_excused(self):
        # its audio was reopened: everyone plays the gap as silence (holes, an underrun, a deficit), but
        # what comes after it is as early as ever - no reason to make anyone wait longer
        p = Player()
        self.run_for(20, self.reports(a=p))
        self.policy.excuse(self.t)
        self.run_for(2, self.reports(a=p))
        p.underruns, p.holes, p.deficit, p.need = 1, 38400, 0.8, 1.2            # the gap reaches the players
        reqs = self.run_for(3, self.reports(a=p))
        p.holes, p.deficit, p.need = 2 * 38400, 0.0, 0.25                       # (holes counted a report later)
        reqs += self.run_for(20, self.reports(a=p))
        self.assertEqual(reqs, [])
        p.underruns += 1                                                        # a real one, later
        self.assertEqual([r.mode for _, r in self.run_for(1, self.reports(a=p))], ["urgent"])

    def test_a_player_who_puts_airpods_in_later_gets_their_room_too(self):
        # joined on its speakers, AirPods in a minute later (the output follows): judged again, like a
        # newcomer, instead of riding out Wi-Fi stalls with 140 ms less for the rest of the session
        p = Player(need=0.031 + 0.0257 + 0.055, lead=0.031)
        self.assertEqual(self.run_for(60, self.reports(a=p)), [])
        p.need, p.lead = AIRPODS_LEAD + 0.0257 + 0.055, AIRPODS_LEAD
        reqs = self.run_for(60, self.reports(a=p))
        self.assertEqual([(r.mode, round(self.schedule.d, 2)) for _, r in reqs], [("join", 0.59)])
        p.lead += 0.01                                                  # (a few ms more is no reason)
        self.assertEqual(self.run_for(30, self.reports(a=p)), [])

    def test_raise_on_10_s_of_reports_under_the_current_delay(self):
        p = Player(need=0.41)                                                   # 40 ms spare: 20 short of MARGIN
        reqs = self.run_for(60, self.reports(a=p))
        self.assertEqual([(t, r.mode, round(r.delta, 3)) for t, r in reqs], [(10.0, "raise", 0.02)])
        self.assertEqual(self.schedule.d, 0.47)

    def test_reports_from_before_a_change_are_not_raised_on(self):
        p = Player(need=0.41)                                                   # 40 ms spare at 0.45
        self.assertEqual(self.run_for(5, self.reports(a=p)), [])                # (not yet 10 reports)
        self.apply(sa.Request("join", 0.14))                                    # 0.59: 180 ms spare from now
        # a player's report lags the change by a second or two: these still show the old 40 ms
        stale = [p.report(0.45) for _ in range(4)]
        reqs = self.run_for(30, lambda: {"a": stale.pop(0) if stale else p.report(self.schedule.d)})
        self.assertEqual(reqs, [])

    def test_lower_only_after_a_calm_5_minutes_and_no_raise_for_10(self):
        self.schedule.d = 0.9
        reqs = self.run_for(900, self.reports(a=Player(need=0.5)))             # 400 ms spare
        self.assertEqual([(t, round(r.delta, 3)) for t, r in reqs], [(300.0, -0.1), (600.0, -0.1)])
        self.assertAlmostEqual(self.schedule.d, 0.7)                            # 200 ms spare: enough is enough
        self.schedule.d, self.t = 0.9, 5000.0
        p = Player(need=0.5)
        self.run_for(1, self.reports(a=p))
        p.underruns = 1                                                         # ran dry: an urgent raise
        self.assertEqual(self.run_for(1, self.reports(a=p))[0][1].mode, "urgent")
        reqs = self.run_for(599, self.reports(a=p))
        self.assertEqual(reqs, [])                                              # no lowering within 10 min

    def test_lowering_stops_at_the_floor(self):
        self.schedule.d = 0.45
        airpods = Player(need=0.05, lead=AIRPODS_LEAD)                          # (an unrealistically fast one)
        reqs = self.run_for(2000, self.reports(a=airpods))
        self.assertEqual([r.mode for _, r in reqs], ["join", "lower", "lower", "lower"])
        self.assertAlmostEqual(self.schedule.d, self.policy.d_min(), places=6)  # ~0.35, and no further
        self.assertAlmostEqual(self.policy.d_min(), AIRPODS_LEAD + self.policy.capture_est + 0.12)

    def test_never_past_2_s(self):
        self.schedule.d = 1.95
        p = Player(need=1.9)
        self.run_for(1, self.reports(a=p))
        p.need, p.deficit, p.underruns = 2.8, 0.9, 1
        (t, r), = self.run_for(1, self.reports(a=p))
        self.assertAlmostEqual(r.delta, 0.05)

    def test_nothing_else_while_a_change_is_pending_but_urgent(self):
        self.schedule.request(sa.Request("raise", 0.1), self.t)
        self.assertTrue(self.schedule.busy)
        p = Player(need=0.41, lead=AIRPODS_LEAD)                                 # would join, and raise
        for _ in range(30):
            self.t += 1
            self.assertIsNone(self.policy.update(self.t, self.reports(a=p)()))
        p.underruns = 1
        self.assertEqual(self.policy.update(self.t + 1, self.reports(a=p)()).mode, "urgent")

    def lower_pending(self, music, seconds=1000, worse_at=None, joins_at=None):
        """Speakers everywhere at 0.45 s (plenty spare: a lower is asked for after 5 minutes), every
        request going through the schedule, which stamps blocks of `music(second)` (loud or silent)."""
        sch = self.schedule
        n, self.t = 0, 1000.0
        here, friend = Player(need=0.12, lead=0.031), Player(need=0.12, lead=0.031)
        players = {"here": here, "friend": friend}
        out = []
        for second in range(seconds):
            for _ in range(100):
                self.t += BLOCK_S
                sch.stamp(n, self.t, 0.5 if music(second) else 0.0, BLOCK_S, self.t)
                n += BLOCK
            if second == worse_at:
                friend.need = sch.d - 0.045             # its Wi-Fi got worse: 45 ms to spare, nothing lost
            if second == joins_at:
                players["airpods"] = Player(need=AIRPODS_LEAD + 0.25, lead=AIRPODS_LEAD)
            req = self.policy.update(self.t, {k: p.report(sch.d) for k, p in players.items()})
            if req:
                sch.request(req, self.t)
                out.append((second, req.mode))
        return out

    def test_a_lower_waiting_for_a_silence_gives_way(self):
        # gapless music: the lower waits (up to 10 minutes); meanwhile the Wi-Fi gets worse, then a
        # friend joins on AirPods - both are seen to at once, and the lower is given up
        out = self.lower_pending(lambda s: True, 360, worse_at=310)
        self.assertEqual(out[0], (300, "lower"))
        self.assertEqual(out[1][1], "raise")
        self.assertLessEqual(out[1][0], 330)
        self.assertFalse(self.schedule.lowering)
        self.setUp()
        out = self.lower_pending(lambda s: True, 320, joins_at=305)
        self.assertEqual(out, [(300, "lower"), (305, "join")])

    def test_a_lower_waiting_for_a_silence_outlives_a_missed_report(self):
        # the host's 1 Hz tick runs a little over a second and the listeners' STATS a little under: now
        # and then a tick sees no new report. That is no reason to give up a lower
        events = []
        sch = self.schedule = sa.DelaySchedule(0.59, lambda e, **f: events.append(e))
        self.policy = sa.DelayPolicy(sch)
        here, friend = Player(need=0.10), Player(need=0.20)
        last = None
        for i in range(1500):
            self.t += 1.005
            rep = friend.report(sch.d) if i % 97 or last is None else last
            last = rep
            req = self.policy.update(self.t, {"here": here.report(sch.d), "friend": rep})
            if req:
                self.assertEqual(req.mode, "lower")
                sch.request(req, self.t)                # (gapless music: it waits for its silence)
        self.assertTrue(sch.lowering)
        self.assertNotIn("lower_abandoned", events)

    def test_a_lower_is_not_applied_on_old_reports(self):
        # the lower asked for at 300 s finds its silence only at 500 s, after the Wi-Fi got worse: it
        # was dropped by then (else the friend would run dry, and an urgent step follow)
        out = self.lower_pending(lambda s: s != 500, 520, worse_at=310)
        self.assertNotIn("urgent", [mode for _, mode in out], out)
        self.assertGreaterEqual(self.schedule.d, 0.45)
        self.assertFalse(self.schedule.lowering)


class CaptureGapTest(unittest.TestCase):
    """The host reopens its audio (its output followed AirPods): capture stops for up to a second and
    carries on in real time, so every player plays a gap of silence - holes, an underrun, a deficit -
    though the audio after it arrives as early as ever. In virtual time: an AirPods listener's real
    engine, the host's schedule stamping each block and its policy asked once a second."""

    def test_a_gap_in_the_hosts_capture_leaves_the_delay_alone(self):
        import vdevice as vd
        from test_output_engine import Mac
        for gap0, gap in ((6.0, 0.5), (6.3, 0.8), (6.6, 1.2)):
            sim = vd.Sim()
            m = Mac(sim, 30, frames=2048, lead=0.2)
            T0 = sim.t
            sch = sa.DelaySchedule(0.59)
            policy = sa.DelayPolicy(sch)
            stamps, k, reported = {}, [0], [0]

            def capture():
                n = k[0] * vd.BLOCK
                if not (gap0 <= n / RATE < gap0 + gap):
                    if gap0 + gap <= n / RATE < gap0 + gap + BLOCK_S:
                        policy.excuse(sim.t)                # (HostSession, once capture is back)
                    stamps[n] = sch.stamp(n, T0 + n / RATE, 0.5, BLOCK_S, sim.t)
                    sim.at(sim.t + 0.012, lambda n=n: m.sb.feed(n, stamps[n], vd.signal(n, vd.BLOCK)))
                k[0] += 1
                sim.at(T0 + (k[0] + 1) * BLOCK_S, capture)
            sim.at(T0 + BLOCK_S, capture)

            def second():
                if gap0 <= sim.t - T0 < gap0 + gap + BLOCK_S:
                    return                                  # (HostSession's tick thread is reopening)
                rep = m.engine.second()
                reported[0] += rep["spare_1s_ms"] is not None
                req = policy.update(sim.t, {"friend": {**rep, "holes": m.sb.holes, "reported_s": reported[0]}})
                if req:
                    sch.request(req, sim.t)
            sim.every(1.0, second, first=T0 + 1.0)
            sim.run(30)
            m.close()
            self.assertGreater(m.sb.holes, 0)
            self.assertEqual(sch.d, 0.59, (gap0, gap))


class CaptureStallTest(unittest.TestCase):
    """BlackHole itself stops calling back for a while (coreaudiod busy, say): under the capture
    watchdog's second, or until it reopens the device. The same: a gap for every player, and nothing a
    longer delay would help. The session excuses it as soon as its capture is late, whenever in the
    second that is (HostSession._watch_capture, AudioInput.missed). In virtual time: the host's real
    capture on a virtual BlackHole, its schedule and policy, and an AirPods listener's engine."""

    def run_gap(self, gap, phase):
        import audio_engine as ae
        import vdevice as vd
        from test_output_engine import Mac
        sim = vd.Sim()
        m = Mac(sim, 22, frames=2048, lead=0.2)
        T0 = sim.t
        sch = sa.DelaySchedule(0.59)
        host = sa.HostSession(None, "Den", log=lambda *a: None)
        host.policy = sa.DelayPolicy(sch)
        host._reopen_input = lambda: None           # (here it comes back by itself)

        def on_block(pcm, n, captured_at):
            at = sch.stamp(n, captured_at, float(np.abs(pcm).max()), BLOCK_S, sim.t)
            sim.at(sim.t + 0.012, lambda: m.sb.feed(n, at, pcm))
        sd = vd.VirtualSD(sim, [{"name": "BlackHole 2ch", "max_input_channels": 2, "lead": 0.005, "seconds": 22}])
        host.source = ae.AudioInput(sd, 0, RATE, 2, on_block, now=lambda: sim.t, threads=False)
        sim.every(ae.CaptureEngine.PERIOD, host.source.capture.tick)
        sim.at(T0 + 6.3, lambda: host.source.stream.device.drop(int(gap * RATE)))
        reported, reqs = [0], []

        def second():
            rep = m.engine.second()
            reported[0] += rep["spare_1s_ms"] is not None
            host._watch_capture(sim.t)                  # (in _second, before the policy)
            req = host.policy.update(sim.t, {"friend": {**rep, "holes": m.sb.holes, "reported_s": reported[0]}})
            if req:
                reqs.append((round(sim.t - T0, 1), req.mode))
                sch.request(req, sim.t)
        sim.every(1.0, second, first=T0 + 1.0 + phase)
        sim.run(22)
        m.close()
        host.source.close()
        return m, reqs

    def test_a_capture_gap_is_excused_whenever_the_reports_come(self):
        for gap in (0.6, 1.5):
            for phase in (0.0, 0.3, 0.6, 0.9):
                m, reqs = self.run_gap(gap, phase)
                self.assertGreater(m.engine.counts["underrun_events"], 0, (gap, phase))    # it did run dry
                self.assertEqual(reqs, [], (gap, phase))


class ScheduleTest(unittest.TestCase):
    """DelaySchedule.stamp on blocks of BLOCK samples, 10 ms apart."""

    def setUp(self):
        self.events = []
        self.s = sa.DelaySchedule(0.45, lambda kind, **f: self.events.append((kind, f)))
        self.n, self.t = 0, 1000.0

    def blocks(self, seconds, loud=True):
        """Stamps of `seconds` of blocks: [(n, play_at)]."""
        out = []
        for _ in range(round(seconds / BLOCK_S)):
            captured = self.t
            out.append((self.n, self.s.stamp(self.n, captured, 0.5 if loud else 0.0, BLOCK_S, self.t)))
            self.n += BLOCK
            self.t += BLOCK_S
        return out

    def delays(self, stamps):
        return np.array([at - (1000.0 + n / RATE) for n, at in stamps])

    def changes(self):
        return [(f["mode"], f["to_ms"], f["from_n"]) for kind, f in self.events if kind == "delay_change"]

    def test_a_raise_waits_for_a_gap_between_songs(self):
        self.s.request(sa.Request("raise", 0.1), self.t)
        loud = self.blocks(2.0)
        quiet = self.blocks(1.0, loud=False)
        self.assertTrue(np.allclose(self.delays(loud), 0.45))
        d = self.delays(quiet)
        first = int(np.argmax(d > 0.46))
        self.assertEqual(first, 29)                                     # the block that ends 0.3 s of silence
        self.assertTrue(np.allclose(d[first:], 0.55) and np.allclose(d[:first], 0.45))
        self.assertEqual(self.changes(), [("silence", 550.0, quiet[first][0])])
        self.assertFalse(self.s.busy)

    def test_no_gap_in_5_s_glides(self):
        self.s.request(sa.Request("raise", 0.1), self.t)
        stamps = self.blocks(50.0)
        d = self.delays(stamps)
        steps = np.diff(d)
        self.assertTrue(np.allclose(d[:500], 0.45))                     # waited 5 s for a gap
        self.assertLessEqual(steps.max(), 0.0025 * BLOCK_S + 1e-12)     # 25 us per 10 ms block: 0.25 % slow
        self.assertGreaterEqual(steps.min(), -1e-9)
        self.assertAlmostEqual(d[-1], 0.55)
        self.assertAlmostEqual(np.argmax(d >= 0.55 - 1e-9) * BLOCK_S, 5 + 40, delta=0.05)   # 100 ms in 40 s
        self.assertEqual([kind for kind, _ in self.events], ["delay_change", "glide_done"])
        self.assertEqual(self.events[0][1]["mode"], "glide")

    def test_join_and_manual_step_at_their_deadline_without_a_gap(self):
        for mode, wait in (("join", 5.0), ("manual", 3.0)):
            self.events.clear()
            self.s.request(sa.Request(mode, 0.05), self.t)
            d = self.delays(self.blocks(wait + 1.0))
            first = int(np.argmax(np.abs(d - d[0]) > 0.01))
            self.assertAlmostEqual(first * BLOCK_S, wait, delta=0.011)
            self.assertEqual(self.changes()[0][0], mode)                # a short faded step, on every Mac alike

    def test_urgent_is_at_once_and_ends_a_glide(self):
        self.s.request(sa.Request("raise", 0.2), self.t)
        self.blocks(6.0)                                                # gliding
        self.assertIsNotNone(self.s.glide_to)
        self.s.request(sa.Request("urgent", 0.05), self.t)
        before = self.s.d
        self.blocks(0.01)
        self.assertAlmostEqual(self.s.d, before + 0.05)
        self.assertIsNone(self.s.glide_to)
        self.assertEqual(self.changes()[-1][0], "urgent")

    def test_a_lower_needs_a_silence_longer_than_it_removes(self):
        self.s.d = 0.6
        self.s.request(sa.Request("lower", -0.1), self.t)
        self.blocks(1.0)
        self.blocks(0.45, loud=False)                                   # not long enough (0.5 s)
        self.blocks(1.0)
        self.assertEqual(self.changes(), [])
        quiet = self.blocks(0.6, loud=False)
        d = self.delays(quiet)
        first = int(np.argmax(d < 0.59))
        self.assertEqual(first, 49)                                     # after 0.5 s of silence
        self.assertEqual(self.changes(), [("silence", 500.0, quiet[first][0])])

    def test_a_lower_waits_quietly_and_anything_else_replaces_it(self):
        self.s.d = 0.6
        self.s.request(sa.Request("lower", -0.1), self.t)
        self.assertEqual((self.s.busy, self.s.lowering, self.s.target), (False, True, 0.6))    # (nothing to show)
        self.s.request(sa.Request("join", 0.05), self.t)
        self.assertEqual((self.s.busy, self.s.lowering, self.s.target), (True, False, 0.65))
        self.s.request(sa.Request("lower", -0.1), self.t)
        self.s.drop_lower("not enough spare any more")
        self.assertEqual((self.s.pending, self.events[-1][0]), (None, "lower_abandoned"))

    def test_a_lower_with_no_long_silence_is_given_up(self):
        self.s.d = 0.6
        self.s.request(sa.Request("lower", -0.1), self.t)
        self.blocks(601.0)
        self.assertEqual(self.events, [("lower_abandoned", {"by_ms": -100.0, "reason": ""})])
        self.assertEqual(self.s.d, 0.6)
        self.assertFalse(self.s.busy)


class HostStampTest(unittest.TestCase):
    """Host.on_block: each block's stamp from the schedule, a change announced with the sample it starts at."""

    def test_stamps_and_announces(self):
        host = sa.Host(RATE, 2, 0.45, port=0, log=lambda *a: None, now=lambda: clock[0], stamps=True)
        sent = []
        peer = type("Peer", (), {"old": False, "codec": None, "push": lambda self, data, *a: sent.append(data)})()
        host.peers.add(peer)
        clock = [1000.0]
        pcm = np.zeros((BLOCK, 2), np.float32)
        for k in range(100):
            if k == 50:
                host.schedule.request(sa.Request("urgent", 0.1), clock[0])
            host.on_block(pcm, k * BLOCK, 990.0 + k * BLOCK_S)
            clock[0] += BLOCK_S
        stamps = list(host.stamps)
        self.assertEqual(stamps[49], (49 * BLOCK, 990.0 + 49 * BLOCK_S + 0.45))
        self.assertAlmostEqual(stamps[50][1], 990.0 + 50 * BLOCK_S + 0.55)
        self.assertEqual(host.delay, host.schedule.d)
        update = [d for d in sent if d[:1] == sa.UPDATE]
        self.assertEqual(len(update), 1)
        import json
        msg = json.loads(update[0][sa.FRAME.size:])
        self.assertEqual((msg["delay"], msg["from_n"], msg["mode"]), (0.55, 50 * BLOCK, "urgent"))
        self.assertEqual([e["event"] for e in host.events], ["delay_change"])
        self.assertTrue(math.isfinite(host.schedule.changed_at))


if __name__ == "__main__":
    unittest.main()
