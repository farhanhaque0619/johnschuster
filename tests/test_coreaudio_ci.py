"""The Mac CI's real-CoreAudio tests (tests/coreaudio), checked where there is no Mac.

- The judge, on recordings made up here with known holes, jumps, stale audio, lateness, speed and
  wobble: it must find exactly those, and time them right.
- The fake BlackHole (tests/coreaudio/fakemac, libsacore's sa_fake_loop): what one stream plays,
  another records at the same sample time, between processes.
- wait_devices.py and the whole of run_all.py on fake devices: every phase runs its processes, judges
  and reports; the canary is caught; the facts are right. Whether a timing phase passes on a shared
  machine is printed, not asserted (on the Mac CI it is the release gate).
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
ca = os.path.join(here, "coreaudio")
sys.path.insert(0, root)
sys.path.insert(0, ca)
import sacore  # noqa: E402
import judge  # noqa: E402
from syncaudio import ClickSource  # noqa: E402

RATE = judge.RATE
has_loop = sacore.lib is not None and hasattr(sacore.lib, "sa_fake_loop")


def code(n):
    """The right channel for (fractional) sample n, as an ideal resampler plays it."""
    return ((np.asarray(n) % judge.CODE) - judge.CODE // 2) / judge.CODE


def content(n):
    n = np.asarray(n, np.float64)
    return np.stack([judge.LEVEL * np.sin(2 * np.pi * judge.SINE_HZ * n / RATE), code(n)], axis=1)


def recording(pcm, t0=1001.0, wobble=0.0, block=512, rate=RATE):
    """What recorder.py would have written: pcm from device frame 0, captured from host time t0 on,
    with an ADC stamp per 512-frame callback (wobbling by up to `wobble` s) from a device whose clock
    really runs at `rate`."""
    k = np.arange(0, len(pcm), block)
    recs = np.zeros(len(k), sacore.LOG_DTYPE)
    recs["seq"], recs["pos"], recs["frames"], recs["valid"] = np.arange(1, len(k) + 1), k, block, block
    recs["dev_time"] = t0 + k / rate + np.random.default_rng(1).uniform(-wobble, wobble, len(k))
    recs["current_time"] = t0 + k / rate + (block + 600) / rate
    return judge.Recording(None, pcm=np.asarray(pcm, np.float32), recs=recs, meta={"skips": 0})


class JudgeTest(unittest.TestCase):
    T0, D = 1000.0, 0.5

    def played(self, seconds=4.0, late=0.0, speed=1.0, t0=1001.0):
        """Sample index heard at each recorded frame: on time is (t - T0 - D) * rate."""
        t = t0 + np.arange(int(seconds * RATE)) / RATE
        due = (t - self.T0 - self.D - late) * RATE
        return due[0] + (due - due[0]) * speed

    def test_steady_and_late_by_a_known_amount(self):
        rec = recording(content(self.played(late=0.0003)))
        r = judge.ramp(rec, self.T0, self.D, 0.0, 1001.2, 1004.8)
        self.assertEqual((r["holes"], r["discontinuities"]), (0, 0), r["events"])
        self.assertAlmostEqual(r["err_median_ms"], 0.3, places=3)
        self.assertLess(r["err_p99_ms"] - r["err_p1_ms"], 0.001)
        self.assertAlmostEqual(r["drift_ms"], 0.0, places=3)
        self.assertAlmostEqual(r["gain"], 1.0, places=4)

    def test_finds_a_hole_a_jump_and_stale_audio_exactly(self):
        n = self.played()
        pcm = content(n)
        pcm[48000:50400] = 0                                # 50 ms of silence
        pcm[96000:] = content(n[96000:] + 5)                # 5 samples skipped: plays 0.104 ms early after
        pcm[144000:145024] = content(n[144000:145024] - 65536)     # what BlackHole's ring held a lap ago
        r = judge.ramp(recording(pcm), self.T0, self.D, 0.0, 1001.2, 1004.8)
        ev = r["events"]
        self.assertEqual([e["kind"] for e in ev], ["hole", "discontinuity", "discontinuity"], ev)
        self.assertAlmostEqual(ev[0]["t"], 1002.0, places=3)
        self.assertAlmostEqual(ev[0]["ms"], 50.0, delta=0.1)
        self.assertAlmostEqual(ev[1]["t"], 1003.0, places=3)
        self.assertAlmostEqual(ev[2]["t"], 1004.0, places=3)
        self.assertAlmostEqual(ev[2]["ms"], 1024 / RATE * 1000, delta=0.1)
        self.assertAlmostEqual(r["drift_ms"], -5 / RATE * 1000, places=3)

    @unittest.skipUnless(sacore.lib is not None, "needs libsacore")
    def test_speed_nudges_and_the_codes_wrap_are_not_glitches(self):
        """Played through libsacore's resampler at the engine's fastest nudge, which smears every wrap of
        the code over +-16 frames."""
        n = self.played(seconds=6.0, speed=1.003)
        src = sacore.Content(2, RATE, seconds=16.0)
        base = int(n[0]) - 64
        src.put(base, content(np.arange(base, base + 7 * RATE)).astype(np.float32))
        pcm = np.zeros((len(n), 2), np.float32)
        self.assertEqual(src.resample(float(n[0]), 1.003, len(n), pcm), 0)
        src.close()
        r = judge.ramp(recording(pcm), self.T0, self.D, 0.0, 1001.0, 1007.0)
        self.assertEqual(r["events"], [])
        self.assertLess(r["drift_ms"], -10.0)               # but measured: 0.3 % fast runs early

    def test_stream_latency_is_taken_back(self):
        """A twin with 80 ms of stream latency plays content n 80 ms before play_at(n) at its loopback."""
        rec = recording(content(self.played(late=-0.080)))
        r = judge.ramp(rec, self.T0, self.D, 0.080, 1001.2, 1004.8)
        self.assertAlmostEqual(r["err_median_ms"], 0.0, places=3)

    def test_recorder_timestamps_through_the_dll(self):
        rec = recording(np.zeros((4 * RATE, 2)))
        self.assertLess(np.abs(rec.t - (1001.0 + np.arange(4 * RATE) / RATE)).max(), 1e-9)
        fast = recording(np.zeros((8 * RATE, 2)), rate=RATE * (1 + 300e-6))      # a device 300 ppm fast
        self.assertLess(np.abs(fast.t - (1001.0 + np.arange(8 * RATE) / (RATE * (1 + 300e-6))))[4 * RATE:].max(), 2e-5)
        wobbly = recording(np.zeros((8 * RATE, 2)), wobble=0.001)
        later = slice(4 * RATE, None)
        self.assertLess(np.abs(wobbly.t[later] - (1001.0 + np.arange(8 * RATE)[later] / RATE)).max(), 2e-4)

    def test_an_empty_window_is_said_so(self):
        r = judge.ramp(recording(content(self.played())), self.T0, self.D, 0.0, 2000.0, 2001.0)
        self.assertIn("error", r)


def clicks_content(n):
    """ClickSource(tone=True) at (fractional) sample n: the tone on the left, its clicks on the right."""
    n = np.asarray(n, np.float64)
    out = np.zeros((len(n), 2))
    out[:, 0] = ClickSource.TONE_LEVEL * np.sin(2 * np.pi * ClickSource.TONE_HZ * n / RATE)
    phase = n - np.floor(n / judge.BEAT) * judge.BEAT
    f = np.where((np.floor(n / judge.BEAT) % 4) == 0, 2000.0, 1000.0)
    tt = phase / RATE
    hit = phase < 0.015 * RATE
    out[hit, 1] = np.sin(2 * np.pi * f[hit] * tt[hit]) * np.exp(-tt[hit] / 0.003) * 0.7
    return out


class ClicksTest(unittest.TestCase):
    T0, D = 1000.0, 0.45

    def stamps(self, until=20 * RATE):
        n = np.arange(0, until, 480)
        return np.stack([n, self.T0 + n / RATE + self.D], axis=1)

    def heard(self, late, seconds=8.0, speed=None):
        t = 1001.0 + np.arange(int(seconds * RATE)) / RATE
        n = (t - self.T0 - self.D - late) * RATE
        if speed:                                   # (from, to, speed): a glide in the middle
            a, b, s = speed
            inside = np.clip(np.arange(len(n)) - a, 0, b - a)
            n = n + inside * (s - 1)
        return n

    def test_click_onsets_against_the_stamps_to_a_fraction_of_a_sample(self):
        late = 59.5 / RATE                                  # half a sample from any whole one
        rec = recording(clicks_content(self.heard(late)))
        r = judge.clicks(rec, self.stamps(), 0.0, 1001.5, 1008.5)
        self.assertGreaterEqual(r["clicks"], 13)
        self.assertEqual(r["clicks_missing"], 0)
        self.assertAlmostEqual(r["err_median_ms"], late * 1000, delta=0.004)
        self.assertLess(r["err_max_ms"] - late * 1000, 0.004)
        self.assertEqual(r["events"], [])

    def test_a_glide_is_continuous_but_a_jump_or_a_gap_is_not(self):
        n = self.heard(0.0, speed=(2 * RATE, 4 * RATE, 1.0025))
        self.assertEqual(judge.clicks(recording(clicks_content(n)), self.stamps(), 0.0, 1001.5, 1008.5)["events"], [])
        pcm = clicks_content(n)
        pcm[5 * RATE:] = clicks_content(n[5 * RATE:] + 1)       # one sample skipped
        pcm[6 * RATE:6 * RATE + 480] = 0                         # 10 ms of silence
        ev = judge.clicks(recording(pcm), self.stamps(), 0.0, 1001.5, 1008.5)["events"]
        self.assertEqual([e["kind"] for e in ev], ["discontinuity", "hole"], ev)
        self.assertAlmostEqual(ev[0]["t"], 1006.0, delta=0.011)     # (to its 5 ms windows)
        self.assertAlmostEqual(ev[1]["t"], 1007.0, delta=0.011)


class BetweenTest(unittest.TestCase):
    def test_two_devices_apart_by_a_known_amount_one_through_the_16_bit_wire(self):
        t0, lat_a = 1001.0, 0.080
        n0 = 5 * judge.CODE + 1234.37                           # any sample, a fraction of one in
        frames = np.arange(6 * RATE)
        a = content(n0 + frames)                                # a: recorded 80 ms before its ear time
        nb = n0 + frames - RATE * (lat_a + 0.0007)              # b: plays 0.7 ms later than a at the ear
        b = content(nb)
        b[:, 1] = code(np.round(nb))                            # the 16-bit wire rounds the code to a sample
        r = judge.between(recording(a, t0), lat_a, recording(b, t0), 0.0, t0 + 1.0, t0 + 5.0, b_wire=True)
        self.assertEqual((r["a"]["events"], r["b"]["events"]), ([], []))
        self.assertAlmostEqual(r["err_median_ms"], -0.7, delta=0.011)      # (+ = a later)
        self.assertGreater(r["compared_s"], 3.5)
        b[2 * RATE:2 * RATE + 24000] = 0                         # 0.5 s of nothing (> the code's period)
        r = judge.between(recording(a, t0), lat_a, recording(b, t0), 0.0, t0 + 1.0, t0 + 5.0, b_wire=True)
        self.assertEqual([e["kind"] for e in r["b"]["events"]], ["hole"], r["b"])
        self.assertAlmostEqual(r["err_median_ms"], -0.7, delta=0.011)      # still in step after it
        b[3 * RATE:] = content(nb[3 * RATE:] + 3)               # b jumps 3 samples
        r = judge.between(recording(a, t0), lat_a, recording(b, t0), 0.0, t0 + 1.0, t0 + 5.0, b_wire=True)
        self.assertEqual(r["b"]["discontinuities"], 1, r["b"])


@unittest.skipUnless(has_loop, "needs the test build of libsacore")
class FakeLoopbackTest(unittest.TestCase):
    """The fake BlackHole: what an output stream plays comes back on an input stream at the same
    sample time, with ADC time == the DAC time it was played at."""

    def test_output_comes_back_at_its_own_sample_time(self):
        state = tempfile.mkdtemp()
        prog = f"""
import sys, time, json, numpy as np
sys.path[:0] = [{os.path.join(ca, 'fakemac')!r}, {root!r}]
import sounddevice as sd, sacore
out, inp = sacore.Ring("output", 2, 48000), sacore.Ring("input", 2, 48000)
kw = dict(samplerate=48000, device="BlackHole 2ch", channels=2, dtype="float32", latency=0.0)
so = sd._StreamBase(kind="output", blocksize=1024, callback=out.callback, userdata=out.h, **kw)
si = sd._StreamBase(kind="input", blocksize=512, callback=inp.callback, userdata=inp.h, **kw)
si.start(); so.start()
n = np.arange(96000)
out.write(4096, np.stack([np.full(96000, 0.5), ((n % 32768) - 16384) / 32768], axis=1), 0)
time.sleep(2.6)
so.abort(); si.abort()
o, i = out.new_records(), inp.new_records()
pcm = inp.read(0, int(inp.state()["w"]))
dac = o["dev_time"][0] + (4096 - o["pos"][0]) / 48000      # when output frame 4096 played
q = int(round(i["pos"][0] + (dac - i["dev_time"][0]) * 48000))  # the input frame captured then
got = pcm[q:q + 96000 - 4096]
print(json.dumps({{"q": q, "first": int(np.flatnonzero(pcm[:, 0])[0]), "same": bool(np.array_equal(
    got[64:, 1], (((n[64:len(got)]) % 32768) - 16384) / 32768)), "fake": [so.stats, si.stats]}}))
"""
        run = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, timeout=60,
                             env={**os.environ, "SYNCAUDIO_FAKE_MAC": state})
        self.assertEqual(run.returncode, 0, run.stderr)
        r = json.loads(run.stdout)
        if any(s["late"] for s in r["fake"]):
            self.skipTest(f"the machine stalled a fake IO thread: {r['fake']}")
        self.assertTrue(r["q"] <= r["first"] < r["q"] + 64, r)        # (a fade-in may start at 0)
        self.assertTrue(r["same"], r)


@unittest.skipUnless(has_loop, "needs the test build of libsacore")
class ScriptsOnFakeDevicesTest(unittest.TestCase):
    def env(self, state, twins):
        return {**os.environ, "PYTHONPATH": os.path.join(ca, "fakemac") + os.pathsep + root,
                "SYNCAUDIO_FAKE_MAC": state, "SYNCAUDIO_FAKE_TWINS": "1" if twins else "0"}

    def test_wait_devices_asserts_the_fixture_and_says_whether_the_twins_are_there(self):
        for twins in (True, False):
            state = tempfile.mkdtemp()
            gh, out = os.path.join(state, "gh"), os.path.join(state, "devices.json")
            run = subprocess.run([sys.executable, os.path.join(ca, "wait_devices.py"), "--need", "BlackHole 2ch",
                                  "--want", "SyncTestBT 2ch", "SyncTestSpk 2ch", "--grace", "2", "--json", out,
                                  "--github-output", gh], capture_output=True, text=True, timeout=60,
                                 env=self.env(state, twins))
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            with open(gh) as f:
                self.assertEqual(f.read(), f"twins={'yes' if twins else 'no'}\n")
            with open(out) as f:
                found = {d["name"]: d for d in json.load(f)["devices"]}
            self.assertEqual(found["BlackHole 2ch"]["out"]["safety"], 0)
            if twins:
                self.assertEqual(found["SyncTestBT 2ch"]["out"]["stream_latency"], 3840)
            else:
                self.assertIn("::warning::SyncTestBT 2ch: never appeared", run.stdout)
        run = subprocess.run([sys.executable, os.path.join(ca, "wait_devices.py"), "--need", "No Such 2ch",
                              "--timeout", "3"], capture_output=True, text=True, timeout=60,
                             env=self.env(tempfile.mkdtemp(), False))
        self.assertEqual(run.returncode, 1)
        self.assertIn("No Such 2ch: never appeared", run.stderr)

    @unittest.skipUnless(sys.platform.startswith("linux"), "on a Mac the realtime-audio job runs the real thing")
    def test_every_phase_runs_end_to_end(self):
        """run_all.py on fake BlackHoles at a fifth of the length, without stress (a busy shared
        machine): every phase's processes run and are judged; the facts and the canary must come out
        right; whether the timing phases pass here is printed (on a Mac they are the gate)."""
        work = tempfile.mkdtemp()
        out = os.path.join(work, "results.json")
        started = time.monotonic()
        run = subprocess.run([sys.executable, os.path.join(ca, "run_all.py"), "--fake", "--twins", "yes",
                              "--scale", "0.2", "--stress", "none", "--evidence", work, "--out", out],
                             capture_output=True, text=True, timeout=1500, cwd=work)
        with open(out) as f:
            res = json.load(f)
        phases = res["phases"]
        print(f"\n  run_all on fake devices, {time.monotonic() - started:.0f} s: " +
              ", ".join(f"{n} {p['status']}" for n, p in phases.items()), file=sys.stderr)
        self.assertEqual(list(phases), ["preflight", "P0", "P1", "P2", "P3", "P3app", "P4", "P5", "P6", "P7", "P8a",
                                        "P8b", "P9"], run.stdout[-3000:])
        for name, p in phases.items():
            for a in p.get("attempts", []):
                broke = [c for c in a["checks"] if c["check"].startswith("ran (")]
                self.assertEqual(broke, [], f"{name} didn't run through: {broke}")
        for name in ("preflight", "P0", "P1"):
            self.assertEqual(phases[name]["status"], "pass", (name, phases[name]))
        checks = {c["check"]: c for c in phases["P9"]["attempts"][-1]["checks"]}
        self.assertTrue(checks["the old way got the device's smallest buffer (<= 64 frames)"]["ok"])
        self.assertTrue(checks["2.0 played 2048-frame buffers"]["ok"])
        self.assertEqual(phases["P6"]["status"], "skipped")
        self.assertEqual(run.returncode, 0 if res["ok"] else 1)


if __name__ == "__main__":
    unittest.main()
