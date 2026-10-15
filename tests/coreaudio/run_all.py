"""The Mac CI's release-blocking tests on real CoreAudio (spec 12.4), phase by phase.

Each phase runs in fresh processes: a recorder (recorder.py, an unstressed process capturing the
loopback device through libsacore's C input callback) is started and waited for, then what is tested
plays (engine_play.py, or syncaudio.py itself), then the recorder stops and judge.py checks every
recorded frame against what should have played at that moment. A gating phase that fails runs once
more; a real bug fails twice. A "control" is 2.0 playing 20 s without stress, run first, to measure
what the machine does by itself (a shared VM can drop the odd cycle with no help from us).

  preflight  capture works at all (macOS's microphone permission) and BlackHole passes audio unchanged
  P0         what 2.0 opens: 1024-frame buffers (2048 on the AirPods-like twin), the clock, the device id
  P1         the canary: 1.1.6 under stress MUST fail (it gets the tiny buffer and drops audio), or
             these tests can't see the bug at all
  P2         1.1.6 with 1.1.7's 2048-frame buffer under stress (numbers only)
  P3         2.0 under stress: nothing skipped, run dry, dropped or late
  P4         a forced 60 ms overload: one skip and one hole that long, and no lasting lag - the premise
             of counting skipped cycles in the ring index, checked on a real HAL
  P5         following the Mac's output from one device to another (twins)
  P6         device clock drift: sacore's helper can't reach BlackHole's clock controls yet (skipped)
  P7         the device's rate changed under a running stream (numbers only in 2.0.0)
  P8a        the whole app, host and listener on this Mac, judged at the speaker (PCM, then Opus)
  P8b        host on the AirPods-like twin, listener on the other: in sync with each other (twins)
  P9         `syncaudio selftest` on the AirPods-like twin (twins)

    python tests/coreaudio/run_all.py [--only P3,P4] [--twins yes|no] [--out coreaudio-results.json]
    python tests/coreaudio/run_all.py --fake --scale 0.25 ...   on fake BlackHoles, anywhere (fakemac/)

Evidence goes to ca-<phase>/ (logs, stats, --report files; recordings only of failed attempts).
"""
import argparse
import glob
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(here))
sys.path.insert(0, here)
import judge  # noqa: E402

BH, BT, SPK = "BlackHole 2ch", "SyncTestBT 2ch", "SyncTestSpk 2ch"
PHASES = ["preflight", "P0", "P1", "P2", "P3", "P3app", "P4", "P5", "P6", "P7", "P8a", "P8b", "P9"]
REPORT_ONLY = {"P2", "P6", "P7"}
TWINS_ONLY = {"P5", "P8b", "P9"}
TCC = ("capture is all zeros: TCC denied the microphone to the runner, so macOS hands every recording "
       "silence. See actions/runner-images images/macos/scripts/build/configure-tccdb-macos.sh")
# (sounddevice: with --fake, it is what stands in for sacore.mac.) Prints when the output changed.
FLIP = ("import sys, time, sounddevice, sacore; i = sacore.mac_call('device_info', sys.argv[1], True); "
        "ok = i and sacore.mac_call('set_default_output', i['id']); print(time.time()); sys.exit(0 if ok else 1)")


class Failed(Exception):
    """Something the phase needs didn't work at all (as opposed to a check that failed)."""


class Checks:
    def __init__(self):
        self.items, self.numbers = [], {}

    def __call__(self, what: str, ok, value=None):
        self.items.append({"check": what, "ok": bool(ok), "value": value})

    @property
    def ok(self) -> bool:
        return bool(self.items) and all(c["ok"] for c in self.items)


def allowed(control: int, control_s: float, s: float) -> int:
    """What the machine did by itself in the control, scaled to s seconds."""
    return math.ceil(control * s / control_s) if control else 0


def grew(timeline: list, key: str, t_from: float, t_to: float = math.inf) -> float:
    """How much a cumulative counter in engine_play's timeline grew from t_from to t_to (s after start)."""
    def at(t):
        return next((e[key] for e in reversed(timeline) if e["t"] <= t and e.get(key) is not None), 0)
    return at(t_to) - at(t_from)


def report_lines(path: str) -> list:
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except OSError:
        return []


class Proc:
    """A process under test, its output in a log; stopped with SIGTERM, the way a Mac quits it."""

    def __init__(self, ctx, log: str, cmd: list):
        self.log, self.cmd = log, cmd
        self.f = open(log, "w")
        self.p = subprocess.Popen(cmd, stdout=self.f, stderr=subprocess.STDOUT, cwd=root, env=ctx.env)

    def stop(self, timeout: float = 15.0) -> int:
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(timeout)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.f.close()
        return self.p.returncode

    def wait(self, timeout: float) -> int:
        """Until it ends by itself (killed after timeout s)."""
        try:
            self.p.wait(timeout)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
        self.f.close()
        return self.p.returncode

    def tail(self, n: int = 15) -> str:
        with open(self.log, errors="replace") as f:
            return "".join(f.readlines()[-n:])


class Recorder:
    """recorder.py on one device, from READY to stop(); then .rec is the judge's Recording."""

    def __init__(self, ctx, d: str, device: str, tag: str):
        self.device, self.prefix = device, os.path.join(d, tag)
        self.f = open(self.prefix + ".rec.log", "w")
        self.p = subprocess.Popen([sys.executable, os.path.join(here, "recorder.py"), "--device", device,
                                   "--out", self.prefix], stdout=subprocess.PIPE, stderr=self.f, text=True,
                                  cwd=root, env=ctx.env)
        self.ready = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()
        if not self.ready.wait(20):
            self.stop()
            raise Failed(f"the recorder on {device} never got going (see {self.prefix}.rec.log)")
        self.rec = None

    def _read(self):
        for line in self.p.stdout:
            self.f.write(line)
            self.f.flush()
            if line.strip() == "READY":
                self.ready.set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    def stop(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(30)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        if self.p.returncode or not os.path.exists(self.prefix + ".pcm.npy"):
            raise Failed(f"the recorder on {self.device} failed (see {self.prefix}.rec.log)")
        if self.rec is None:
            self.rec = judge.Recording(self.prefix)
        return self.rec


class Ctx:
    """One run: where the evidence goes, how long phases last, the fake devices or the real ones."""

    def __init__(self, args):
        self.fake, self.twins, self.scale = args.fake, args.twins == "yes", args.scale
        self.base = os.path.abspath(args.evidence)
        self.keep_audio = args.keep_audio
        self.env = {**os.environ, "PYTHONPATH": root, "PYTHONUNBUFFERED": "1"}
        if self.fake:
            lib = "libsacore_test.dylib" if sys.platform == "darwin" else "libsacore.so"
            self.env.update(PYTHONPATH=os.path.join(here, "fakemac") + os.pathsep + root,
                            SYNCAUDIO_FAKE_MAC=tempfile.mkdtemp(prefix="fake-mac-"),
                            SYNCAUDIO_FAKE_TWINS="1" if self.twins else "0",
                            SACORE_LIB=os.environ.get("SACORE_LIB") or os.path.join(root, lib))
        self.load = args.stress
        self.controls: dict = {}
        self.port = 47821

    def stress(self, *args) -> list:
        """The arguments that put a stress on a process (none for a dry run on a busy machine)."""
        return list(args) if self.load != "none" else []

    def s(self, seconds: float) -> float:
        return seconds * self.scale

    def dir(self, name: str) -> str:
        d = os.path.join(self.base, f"ca-{name}")
        os.makedirs(d, exist_ok=True)
        return d

    def play(self, d: str, tag: str, device: str, seconds: float, engine: str = "v2", stress: str = "none",
             *extra) -> dict:
        """engine_play.py to the end; its stats."""
        prefix = os.path.join(d, tag)
        stress = stress if self.load != "none" else "none"
        cmd = [sys.executable, os.path.join(here, "engine_play.py"), "--engine", engine, "--device", device,
               "--seconds", f"{seconds:.2f}", "--stress", stress, "--out", prefix, *map(str, extra)]
        with open(prefix + ".log", "w") as f:
            try:
                run = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=root, env=self.env,
                                     timeout=seconds + 120)
            except subprocess.TimeoutExpired:
                raise Failed(f"engine_play on {device} hung (see {prefix}.log)") from None
        if run.returncode:
            with open(prefix + ".log", errors="replace") as f:
                raise Failed(f"engine_play on {device} failed ({run.returncode}):\n" + "".join(f.readlines()[-15:]))
        with open(prefix + ".json") as f:
            return json.load(f)

    def app(self, d: str, tag: str, *args) -> Proc:
        """syncaudio.py itself (host, join, selftest), its output in d/tag.log."""
        return Proc(self, os.path.join(d, tag + ".log"), [sys.executable, os.path.join(root, "syncaudio.py"),
                                                           *map(str, args)])

    def flip(self, device: str) -> float:
        """Make `device` the Mac's sound output, through sacore's helper (as the app would see AirPods).
        Returns when it changed (time.time(); starting the helper's own Python isn't the app's time)."""
        run = subprocess.run([sys.executable, "-c", FLIP, device], cwd=root, env=self.env, capture_output=True,
                             text=True, timeout=30)
        if run.returncode:
            raise Failed(f"couldn't make {device} the sound output: {run.stderr.strip()[-500:]}")
        return float(run.stdout.split()[-1])

    def keep(self, d: str, ok: bool):
        """Recordings take ~0.4 MB a second: keep them only where something failed."""
        if ok and not self.keep_audio:
            for f in glob.glob(os.path.join(d, "*.npy")):
                os.remove(f)

    def next_port(self) -> int:
        self.port += 2
        return self.port

    def lan_ip(self) -> str:
        """This Mac's own network address (en0), so the listener's audio takes the real network path."""
        if not self.fake:
            for nic in ("en0", "en1"):
                try:
                    ip = subprocess.run(["ipconfig", "getifaddr", nic], capture_output=True, text=True,
                                        timeout=5).stdout.strip()
                except (OSError, subprocess.SubprocessError):
                    ip = ""
                if ip:
                    return ip
        return "127.0.0.1"


def excused(st: dict, *recs) -> tuple:
    """(skipped cycles, heard events) that weren't the engine's doing: the fake IO thread waking late
    (the OS's fault; known only on fake devices) and cycles the recorders' own input skipped (a hole
    in the recording, not in what was played)."""
    late = (st.get("fake") or {}).get("late_wake", 0)
    return late, late + sum(r.meta.get("skips", 0) for r in recs)


def strlat(stats: dict) -> float:
    return stats.get("strlat") or 0.0


def heard_strlat(rec) -> float:
    """The stream latency the player on the recorded device added (BlackHole reports the same both ways)."""
    f = rec.meta.get("facts") or {}
    return f["stream_latency"] / f["nominal_rate"] if f.get("nominal_rate") else 0.0


def must(j: dict) -> dict:
    """A judgement of something: an empty window is a failure, not a pass."""
    if j.get("error") or not j.get("judged_s"):
        raise Failed(f"nothing to judge: {j.get('error', 'an empty window')}")
    return j


def judged(rec, st, warm: float, tail: float = 0.5) -> dict:
    """engine_play's content as recorded, from `warm` s after it started to `tail` s before it stopped."""
    return must(judge.ramp(rec, st["T0"], st["D"], strlat(st), st["started"] + warm, st["stopped"] - tail))


# ------------------------------------------------------------------------------------- phases
def preflight(ctx, d, c):
    with Recorder(ctx, d, BH, "bh") as r:
        st = ctx.play(d, "v2", BH, max(3.0, ctx.s(3.0)))
    rec = r.stop()
    rms = rec.rms(st["started"] + 1.0, st["stopped"] - 0.2)
    played = st["io"][1] is not None
    c("2.0 played into BlackHole 2ch (callbacks came)", played, st["io"])
    if played and rms == 0.0:
        c(TCC, False, rms)
        return
    c("the recording isn't silent (RMS > 0.1)", rms > 0.1, round(rms, 4))
    j = judged(rec, st, 1.0, 0.2)
    c.numbers.update(judge=j)
    c("BlackHole hands back exactly what was played (gain 1: its volume at 100 %)",
      j.get("gain") is not None and abs(j["gain"] - 1) < 1e-3, j.get("gain"))


def p0(ctx, d, c):
    for device in [BH] + ([BT, SPK] if ctx.twins else []):
        tag = device.split()[0]
        st = ctx.play(d, tag, device, max(3.0, ctx.s(3.0)))
        o = st["outputs"][0]
        want = 2048 if device == BT else 1024
        c.numbers[tag] = {"io": st["io"], "plan": o["plan"], "facts": o["facts"], "dev_id": o["dev_id"],
                          "pa_device": o["pa_device"], "clock_delta": st["clock_delta"]}
        c(f"{device}: {want}-frame callbacks, every one", st["io"] == [want, want], st["io"])
        c(f"{device}: planned {want} (Bluetooth-class by its latency: {device == BT})",
          (o["plan"]["blocksize"], o["plan"]["bt_class"]) == (want, device == BT), o["plan"])
        c(f"{device}: CoreAudio's BufferFrameSize reads back {want}",
          (o["facts"] or {}).get("buffer_frames") == want, (o["facts"] or {}).get("buffer_frames"))
        c(f"{device}: libsacore's clock is time.monotonic() within 1 ms", st["clock_delta"] < 1e-3, st["clock_delta"])
        c(f"{device}: the stream's CoreAudio device is {device}", (o["facts"] or {}).get("name") == device,
          [o["dev_id"], (o["facts"] or {}).get("name")])
        if not ctx.fake:        # (the fake has no PortAudio)
            c(f"{device}: PortAudio's PaMacCore_GetStreamOutputDevice resolves it",
              o["pa_device"] is not None and o["pa_device"] == o["dev_id"], [o["pa_device"], o["dev_id"]])


def legacy(ctx, d, c, latency: float, block: int):
    device = BT if ctx.twins else BH
    with Recorder(ctx, d, device, "rec") as r:
        st = ctx.play(d, "legacy", device, ctx.s(30.0), "legacy116", "ci", "--legacy-latency", latency,
                      "--legacy-block", block)
    j = judged(r.stop(), st, 2.0)
    c.numbers.update(device=device, latency=latency, blocksize=block, judge=j,
                     engine={k: st.get(k) for k in ("io", "callbacks", "cb_lost_ms", "pa_flags", "overloads",
                                                    "underflows", "fake")})
    return st, j


def p1(ctx, d, c):
    """Passes when the old engine FAILS, as it must."""
    latency = 0.1 if ctx.twins else 0.0         # both take PortAudio's smallest-buffer branch
    st, j = legacy(ctx, d, c, latency, 0)
    tiny = st["io"] is not None and st["io"][1] <= 64
    dropped = j["holes"] + j["discontinuities"] >= 20 or j["lost_ms"] >= 50 or st["overloads"] >= 20
    c("PortAudio gave 1.1.6's settings the device's smallest buffer (<= 64 frames)", tiny, st["io"])
    c("and 1.1.6 under stress dropped audio (>= 20 holes + jumps, >= 50 ms lost, or >= 20 overloads)", dropped,
      {"holes": j["holes"], "discontinuities": j["discontinuities"], "lost_ms": j["lost_ms"],
       "overloads": st["overloads"]})


def p2(ctx, d, c):
    legacy(ctx, d, c, 0.1, 2048)
    c("ran", True)


def control(ctx, device: str) -> dict:
    """2.0 for 20 s without stress: what this machine does on its own."""
    if device not in ctx.controls:
        d = ctx.dir("control-" + device.split()[0])
        with Recorder(ctx, d, device, "rec") as r:
            st = ctx.play(d, "v2", device, ctx.s(20.0))
        warm = ctx.s(5.0)
        j = judged(r.stop(), st, warm)
        tl = st["timeline"]
        ctx.controls[device] = {
            "window_s": st["stopped"] - st["started"] - warm, "skips": grew(tl, "skips", warm),
            "overloads": grew(tl, "overloads", warm), "events": j["holes"] + j["discontinuities"],
            "excused": excused(st, r.rec), "judge": j, "cpu_pct": st["cpu_pct"]}
        ctx.keep(d, ok=True)
    return ctx.controls[device]


def p3(ctx, d, c, stress="ci", seconds=70.0):
    """2.0 under stress. ci: far beyond anything the app does (a full collection of a 300k-object heap
    every second can hold every Python thread longer than the delay allows on these Macs): the ring may
    run dry only when Python held it up longer than the audio left in it, and rarely. app (P3app): the
    app's own busiest work: nothing may be heard at all."""
    for device in [BH] + ([BT] if ctx.twins else []):
        tag, want = device.split()[0], 2048 if device == BT else 1024
        ctl = control(ctx, device)
        with Recorder(ctx, d, device, tag) as r:
            st = ctx.play(d, tag, device, ctx.s(seconds), "v2", stress)
        warm = ctx.s(10.0)
        j = judged(r.stop(), st, warm)
        tl, window = st["timeline"], st["stopped"] - st["started"] - warm
        ex_skips, ex_events = excused(st, r.rec)
        skips, overloads = grew(tl, "skips", warm), grew(tl, "overloads", warm)
        cb_max = max((e["cb_max_us"] for e in tl if e["t"] > warm), default=None)
        c.numbers[tag] = {"control": ctl, "judge": j, "skips": skips, "overloads": overloads, "cb_max_us": cb_max,
                          "io": st["io"], "excused": [ex_skips, ex_events], "fake": st.get("fake"),
                          "cpu_pct": st["cpu_pct"], "last": tl[-1] if tl else None}
        c(f"{device}: {want}-frame callbacks", st["io"] == [want, want], st["io"])
        c(f"{device}: the machine itself (control) skipped <= 2 cycles", ctl["skips"] <= 2, ctl["skips"])
        c(f"{device}: no more skipped cycles than the control",
          skips <= allowed(ctl["skips"], ctl["window_s"], window) + ex_skips, [skips, ctl["skips"], ex_skips])
        c(f"{device}: no more overloads than the control",
          overloads <= allowed(ctl["overloads"], ctl["window_s"], window), [overloads, ctl["overloads"]])
        last = tl[-1]
        dry = grew(tl, "underruns", warm)
        c(f"{device}: the ring ran dry only when Python held it up longer than the audio in it",
          last.get("unexplained", 0) == 0, [last.get("unexplained"), last["underruns"], last.get("producer_max_ms")])
        if stress == "app":
            c(f"{device}: the ring never ran dry", last["underruns"] == 0, last["underruns"])
        else:
            c(f"{device}: and that cost at most 5 % of the time", last["underrun_ms"] <= 0.05 * 1000 * ctx.s(seconds),
              [last["underrun_ms"], last["underruns"]])
        c(f"{device}: no holes or jumps heard (no more than the control" + (")" if stress == "app" else
                                                                           ", plus one per time it ran dry)"),
          j["holes"] + j["discontinuities"] <= allowed(ctl["events"], ctl["window_s"], window) + ex_events
          + (dry if stress != "app" else 0), [j.get("events"), dry])
        c(f"{device}: the C callback never took 1 ms", cb_max is not None and cb_max <= 1000, cb_max)
        c(f"{device}: steady timing (p99 - p1 <= 0.10 ms)", j["err_p99_ms"] is not None and
          j["err_p99_ms"] - j["err_p1_ms"] <= 0.10, [j["err_p1_ms"], j["err_p99_ms"]])
        if device == BH:            # stock BlackHole has no latency of its own: the timing is absolute
            c(f"{device}: on time (|median| <= 1.0 ms)", abs(j["err_median_ms"]) <= 1.0, j["err_median_ms"])
        c(f"{device}: no drift (<= 0.05 ms from the first 10 s to the last)", abs(j["drift_ms"]) <= 0.05,
          j["drift_ms"])


def p4(ctx, d, c):
    ctl = control(ctx, BH)
    stall_at = ctx.s(20.0)
    with Recorder(ctx, d, BH, "bh") as r:
        st = ctx.play(d, "v2", BH, ctx.s(40.0), "v2", "none", "--stall-at", stall_at, "--stall-us", 60000)
    rec = r.stop()
    warm = ctx.s(5.0)
    j = judged(rec, st, warm)
    tl, window = st["timeline"], st["stopped"] - st["started"] - warm
    ex_skips, ex_events = excused(st, rec)
    stall = st.get("stall", {}).get("t")
    stalled = next((n for n in st["noted"] if n["cb_ms"] >= 50 and stall and n["entry"] >= stall - 0.01), None)
    skips_then = [n for n in st["noted"] if n["skipped_ms"] > 0 and stall and 0 <= n["entry"] - stall < 1.0]
    natural = grew(tl, "skips", 0) - len(skips_then)
    buffer = 1024 / judge.RATE
    holes = [e for e in j["events"] if stalled and abs(e["t"] - (stalled["dac"] - strlat(st))) <= buffer]
    others = [e for e in j["events"] if e not in holes]
    before = must(judge.ramp(rec, st["T0"], st["D"], strlat(st), st["started"] + warm, stall - 1.0)) if stall else {}
    after = must(judge.ramp(rec, st["T0"], st["D"], strlat(st), stall + 1.0, st["stopped"] - 0.5)) if stall else {}
    flags = 0
    for e in tl:
        flags |= e.get("pa_flags") or 0
    c.numbers.update(judge=j, stalled=stalled, skips_then=skips_then, control=ctl,
                     overloads=grew(tl, "overloads", 0), pa_output_underflow=bool(flags & 4),
                     median_before=before.get("err_median_ms"), median_after=after.get("err_median_ms"),
                     fake=st.get("fake"))
    c("the forced 60 ms stall happened in the C callback", stalled is not None, stalled)
    c("libsacore counted exactly one skip for it, of 40-110 ms",
      len(skips_then) == 1 and 40 <= skips_then[0]["skipped_ms"] <= 110, skips_then)
    c("no other skips beyond the control's", natural <= allowed(ctl["skips"], ctl["window_s"], window) + ex_skips,
      [natural, ex_skips])
    c("the recording has exactly one hole there (within a buffer), 40-110 ms long",
      len(holes) == 1 and 40 <= holes[0]["ms"] <= 110, holes)
    c("and nothing else beyond the control's",
      len(others) <= allowed(ctl["events"], ctl["window_s"], window) + ex_events, [others, ex_events])
    c("no lasting lag (median timing moved <= 0.5 ms)", None not in (before.get("err_median_ms"),
      after.get("err_median_ms")) and abs(after["err_median_ms"] - before["err_median_ms"]) <= 0.5,
      [before.get("err_median_ms"), after.get("err_median_ms")])


def clicks_run(ctx, d, tag: str, seconds: float, codec: str, recorders, stress: bool, join_args=(),
               flip_at=None, flip_to=None):
    """syncaudio host --click --tone and syncaudio join on this Mac, while `recorders` record."""
    port = ctx.next_port()
    load = ctx.stress("--stress", "app") if stress else []
    host = ctx.app(d, f"{tag}-host", "host", "--click", "--tone", "--no-local", "--no-switch", "--port", port,
                   "--name", "CI", "--report", os.path.join(d, f"{tag}-host.jsonl"), "--codec", codec, *load)
    time.sleep(2.0)
    join = ctx.app(d, f"{tag}-join", "join", ctx.lan_ip(), "--port", port, "--name", "CI listener",
                   "--report", os.path.join(d, f"{tag}-join.jsonl"), *load, *join_args)
    flipped = None
    try:
        if flip_at is not None:
            time.sleep(flip_at)
            wall = ctx.flip(flip_to)
            flipped = (wall, time.monotonic() - (time.time() - wall))
            time.sleep(seconds - flip_at)
        else:
            time.sleep(seconds)
    finally:
        rc = (join.stop(), host.stop())
    for r in recorders:
        r.stop()
    if rc != (0, 0):
        raise Failed(f"syncaudio exited with {rc}:\n{join.tail()}\n{host.tail()}")
    hrep, jrep = (report_lines(os.path.join(d, f"{tag}-{w}.jsonl")) for w in ("host", "join"))
    if not hrep or not jrep:
        raise Failed("syncaudio wrote no --report lines")
    stamps = [s for line in hrep for s in line.get("stamps", [])]
    return hrep, jrep, stamps, flipped


def steps(hrep: list) -> list:
    return [e for e in hrep[-1]["info"].get("events", []) if e.get("event") == "delay_change"]


def p8a(ctx, d, c):
    for codec, seconds in (("pcm", ctx.s(90.0)), ("auto", ctx.s(30.0))):
        with Recorder(ctx, d, BH, codec) as r:
            hrep, jrep, stamps, _ = clicks_run(ctx, d, codec, seconds, codec, [r], True, ("--output", BH))
        sound = r.rec.sound()
        if not sound:
            c(f"{codec}: anything was heard", False)
            continue
        warm = ctx.s(15.0) if codec == "pcm" else ctx.s(5.0)
        j = must(judge.clicks(r.rec, stamps, heard_strlat(r.rec), sound[0] + warm, sound[1] - 0.5, codec == "auto"))
        last, ch = jrep[-1]["info"], steps(hrep)
        c.numbers[codec] = {"judge": j, "codec": last.get("codec"), "io": last.get("io"), "skips": last.get("skips"),
                            "underruns": last.get("underruns"), "delay_changes": ch, "delay_ms": last.get("delay_ms")}
        c(f"{codec}: clicks heard and timed", j.get("clicks", 0) >= 5, [j.get("clicks"), j.get("clicks_missing")])
        _, ex = excused({}, r.rec)
        c(f"{codec}: no holes or jumps", j["holes"] + j["discontinuities"] <= ex, [j.get("events"), ex])
        if codec == "pcm":
            c("pcm: on the host's stamps (p1, p99 within +-2 ms; |median| <= 1 ms)", j["err_p99_ms"] is not None
              and max(abs(j["err_p1_ms"]), abs(j["err_p99_ms"])) <= 2.0 and abs(j["err_median_ms"]) <= 1.0,
              [j["err_p1_ms"], j["err_median_ms"], j["err_p99_ms"]])
            c("pcm: listener played 1024-frame buffers, none skipped, never ran dry",
              (last.get("io"), last.get("skips"), last.get("underruns")) == ([1024, 1024], 0, 0),
              [last.get("io"), last.get("skips"), last.get("underruns")])
            c("pcm: at most one delay change (the join), none urgent",
              len(ch) <= 1 and not any(e.get("request") == "urgent" for e in ch), ch)
        else:
            c("opus: the audio really went as Opus", last.get("codec") == "opus", last.get("codec"))
            c("opus: every click within 2 ms of its stamp", j["err_max_ms"] is not None and j["err_max_ms"] <= 2.0,
              j["err_max_ms"])


def p8b(ctx, d, c):
    port = ctx.next_port()
    seconds = ctx.s(90.0)
    with Recorder(ctx, d, BT, "bt") as rb, Recorder(ctx, d, SPK, "spk") as rs:
        tone = Proc(ctx, os.path.join(d, "tone.log"), [
            sys.executable, os.path.join(here, "engine_play.py"), "--engine", "v2", "--device", BH,
            "--seconds", f"{seconds + 8:.1f}", "--out", os.path.join(d, "tone")])
        time.sleep(1.0)
        host = ctx.app(d, "host", "host", "--input", BH, "--output", BT, "--no-switch", *ctx.stress("--stress", "app"),
                       "--codec", "pcm", "--port", port, "--name", "CI", "--report", os.path.join(d, "host.jsonl"))
        time.sleep(2.0)
        join = ctx.app(d, "join", "join", "127.0.0.1", "--port", port, "--name", "CI listener", "--output", SPK,
                       *ctx.stress("--stress", "app"), "--report", os.path.join(d, "join.jsonl"))
        try:
            time.sleep(seconds)
        finally:
            rc = (join.stop(), host.stop(), tone.wait(30))
    if rc != (0, 0, 0):
        raise Failed(f"a process exited with {rc}:\n{join.tail()}\n{host.tail()}\n{tone.tail()}")
    a, b = rb.stop(), rs.stop()
    sa, sb = a.sound(), b.sound()
    if not (sa and sb):
        c("both twins heard something", False, [sa, sb])
        return
    la, lb = heard_strlat(a), heard_strlat(b)
    j = judge.between(a, la, b, lb, max(sa[0] + la, sb[0] + lb) + ctx.s(15.0), min(sa[1] + la, sb[1] + lb) - 0.5,
                      b_wire=True)
    if j.get("error") or not j.get("compared_s"):
        raise Failed(f"nothing to compare: {j.get('error', 'no frame good on both')}")
    hrep = report_lines(os.path.join(d, "host.jsonl"))
    cap = (hrep[-1]["info"].get("capture") or {}) if hrep else {}
    ch = steps(hrep) if hrep else []
    c.numbers.update(judge=j, delay_changes=ch, capture={k: v for k, v in cap.items() if k.startswith("cap_")})
    c("the two twins play the same sample together (p1, p99 within +-2 ms)", j.get("err_p99_ms") is not None
      and max(abs(j["err_p1_ms"]), abs(j["err_p99_ms"])) <= 2.0, [j.get("err_p1_ms"), j.get("err_p99_ms")])
    for side, name, rec in (("a", BT, a), ("b", SPK, b)):
        _, ex = excused({}, rec)
        c(f"{name}: no holes or jumps", j[side]["holes"] + j[side]["discontinuities"] <= ex, [j[side]["events"], ex])
    c("at most one delay change", len(ch) <= 1, ch)
    c("capture never skipped or fell behind", (cap.get("cap_skips"), cap.get("cap_overflows")) == (0, 0),
      [cap.get("cap_skips"), cap.get("cap_overflows")])


def p5(ctx, d, c):
    ctx.flip(SPK)
    try:
        with Recorder(ctx, d, SPK, "spk") as rs, Recorder(ctx, d, BT, "bt") as rb:
            seconds, at = ctx.s(35.0), ctx.s(15.0)
            hrep, jrep, stamps, (flip_wall, flip_t) = clicks_run(ctx, d, "follow", seconds, "pcm", [rs, rb], False,
                                                                 flip_at=at, flip_to=BT)
    finally:
        ctx.flip(BH)
    opened = [e for e in jrep[-1]["info"].get("events", []) if e.get("event") == "output_open" and e.get("name") == BT]
    took = round(opened[0]["t"] - flip_wall, 2) if opened else None
    spk, bt = rs.rec.sound(), rb.rec.sound()
    c("the listener started on SyncTestSpk 2ch (the Mac's output)", spk is not None and spk[0] < flip_t, spk)
    # (the session looks every 0.5 s, then: PortAudio's abort (>= 0.1 s), its restart, the open, and the new
    # output's first three callbacks before output_open)
    c("and moved to SyncTestBT 2ch within 1.5 s of the output changing", took is not None and took <= 1.5, took)
    if not (spk and bt):
        return
    js = must(judge.clicks(rs.rec, stamps, heard_strlat(rs.rec), spk[0] + ctx.s(3.0), flip_t + 2.0))
    jb_all = must(judge.clicks(rb.rec, stamps, heard_strlat(rb.rec), bt[0] + 0.01, bt[1] - 0.5))   # (its fade-in)
    jb = must(judge.clicks(rb.rec, stamps, heard_strlat(rb.rec), bt[0] + 3.0, bt[1] - 0.5))
    c.numbers.update(spk=js, bt=jb, bt_whole=jb_all, reopen_s=took, first_sound_after_flip_s=round(bt[0] - flip_t, 3))
    ex_s, ex_b = excused({}, rs.rec)[1], excused({}, rb.rec)[1]
    c("at most one faded gap on SyncTestSpk 2ch (where it stopped)",
      js["holes"] + js["discontinuities"] <= 1 + ex_s, [js["events"], ex_s])
    c("at most one on SyncTestBT 2ch", jb_all["holes"] + jb_all["discontinuities"] <= 1 + ex_b,
      [jb_all["events"], ex_b, steps(hrep), [e for e in jrep[-1]["info"].get("events", [])
                                             if e.get("event") in ("underrun", "output_open", "skip_burst")]])
    c("on SyncTestBT 2ch on time 3 s after the move (every click within 1 ms)",
      jb["err_max_ms"] is not None and jb["err_max_ms"] <= 1.0, jb["err_max_ms"])


def p6(ctx, d, c):
    c.numbers["skipped"] = ("BlackHole's clock source and pitch controls (+500 ppm) aren't reachable through "
                            "sacore's macOS helper; device drift is covered by the virtual-device tests (T2)")


def p7(ctx, d, c):
    at = ctx.s(15.0)                    # 44.1 kHz from here, 48 kHz again 5 s later (engine_play)
    with Recorder(ctx, d, BH, "bh") as r:
        st = ctx.play(d, "v2", BH, at + 5.0 + max(8.0, ctx.s(20.0)), "v2", "none", "--rate-at", at)
    rec = r.stop()
    changes = st.get("rate_changes", [])
    c.numbers.update(rate_changes=changes, actions=st["actions"], outputs=[
        {k: o[k] for k in ("t", "reason", "level", "frames")} for o in st["outputs"]])
    c("the rate was changed and changed back", len(changes) == 2 and all(x["ok"] for x in changes), changes)
    if len(changes) == 2:
        back = changes[1]["t"]
        j = must(judge.ramp(rec, st["T0"], st["D"], strlat(st), back + 2.0, st["stopped"] - 0.5))
        c.numbers["after"] = j
        ex = excused(st, rec)[1]
        c("audio back within 2 s of the rate coming back", j["holes"] + j["discontinuities"] <= ex, [j["events"], ex])
        c("on time after it (<= 1 ms)", j["err_max_ms"] is not None and j["err_max_ms"] <= 1.0, j["err_max_ms"])


def p9(ctx, d, c):
    with open(os.path.join(d, "selftest.log"), "w") as log:
        run = subprocess.run([sys.executable, os.path.join(root, "syncaudio.py"), "selftest", "--output", BT, "--json"],
                             stdout=subprocess.PIPE, stderr=log, text=True, cwd=root, env=ctx.env, timeout=120)
    if run.returncode:
        raise Failed(f"syncaudio selftest exited with {run.returncode} (see {d}/selftest.log)")
    res = json.loads(run.stdout)
    c.numbers.update(result=res)
    lg, v2 = res.get("legacy") or {}, res["v2"]
    c("the old way got the device's smallest buffer (<= 64 frames)", (lg.get("frames_max") or 1e9) <= 64,
      lg.get("frames_max"))
    c("2.0 played 2048-frame buffers", v2["io_frames"] == 2048, v2["io_frames"])
    c("none skipped", v2["skips"] == 0, v2["skips"])
    c("verdict: clean", res["verdict"] == "clean", res["verdict"])


def p3app(ctx, d, c):
    p3(ctx, d, c, "app", 40.0)


RUN = {"preflight": preflight, "P0": p0, "P1": p1, "P2": p2, "P3": p3, "P3app": p3app, "P4": p4, "P5": p5, "P6": p6,
       "P7": p7,
       "P8a": p8a, "P8b": p8b, "P9": p9}


def phase(ctx, name: str) -> dict:
    gate = name not in REPORT_ONLY
    if name in TWINS_ONLY and not ctx.twins:
        return {"gate": gate, "status": "skipped", "why": "the twins aren't installed"}
    if name == "P6":
        c = Checks()
        p6(ctx, None, c)
        return {"gate": False, "status": "skipped", "why": c.numbers["skipped"]}
    attempts = []
    for attempt in (1, 2) if gate else (1,):
        d = ctx.dir(name if attempt == 1 else f"{name}-retry")
        c = Checks()
        started = time.monotonic()
        try:
            RUN[name](ctx, d, c)
        except Exception as e:      # noqa: BLE001 - one phase breaking must not stop the others
            c(f"ran ({type(e).__name__})", False, str(e) if isinstance(e, Failed) else traceback.format_exc())
        attempts.append({"ok": c.ok, "seconds": round(time.monotonic() - started, 1), "dir": os.path.relpath(d),
                         "checks": c.items, "numbers": c.numbers})
        ctx.keep(d, c.ok)
        print(f"  {name} attempt {attempt}: {'pass' if c.ok else 'FAIL'} ({attempts[-1]['seconds']} s)", flush=True)
        for item in c.items:
            print(f"    [{'ok' if item['ok'] else 'XX'}] {item['check']}: {json.dumps(item['value'], default=str)[:300]}",
                  flush=True)
        if c.ok:
            break
    ok = attempts[-1]["ok"]
    out = {"gate": gate, "status": ("pass" if ok else "fail") if gate else "report", "attempts": attempts}
    if name == "P1" and not ok:
        out["why"] = "canary didn't reproduce: the test can't see the bug"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", action="append", help="phases to run, e.g. preflight or P3,P4 (default: all)")
    ap.add_argument("--twins", default="no", help="yes: SyncTestBT 2ch / SyncTestSpk 2ch are installed")
    ap.add_argument("--out", default="coreaudio-results.json")
    ap.add_argument("--evidence", default=".", help="where the ca-<phase> folders go")
    ap.add_argument("--fake", action="store_true", help="fake BlackHoles (tests/coreaudio/fakemac): any machine")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every phase's length (fake runs)")
    ap.add_argument("--keep-audio", action="store_true", help="keep every recording, not only failed ones")
    ap.add_argument("--stress", choices=["ci", "none"], default="ci",   # (ci: P3's ci phase, app elsewhere)
                    help="none: leave out the GIL stress (dry runs on a busy machine; never in CI)")
    args = ap.parse_args(argv)
    only = [p for arg in (args.only or []) for p in arg.split(",") if p]
    unknown = [p for p in only if p not in PHASES]
    if unknown:
        ap.error(f"unknown phase(s) {unknown}; there are {PHASES}")
    ctx = Ctx(args)
    names = [p for p in PHASES if not only or p in only]
    print(f"Real-CoreAudio phases {names} ({'FAKE devices' if ctx.fake else 'this Mac'}, twins {args.twins})",
          flush=True)
    results = {"fake": ctx.fake, "twins": ctx.twins, "scale": ctx.scale, "phases": {}}
    for name in names:
        print(f"== {name} ==", flush=True)
        results["phases"][name] = phase(ctx, name)
        results["controls"] = ctx.controls
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1, default=str)
    failed = [n for n, r in results["phases"].items() if r["gate"] and r["status"] == "fail"]
    results["ok"] = not failed
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("\n" + "\n".join(f"{n:10} {r['status']:8} {r.get('why', '')}" for n, r in results["phases"].items()))
    if failed:
        print(f"\nRelease-blocking phases failed: {failed}" + (
            "\nP1: canary didn't reproduce: the test can't see the bug" if "P1" in failed else ""))
        sys.exit(1)


if __name__ == "__main__":
    main()
