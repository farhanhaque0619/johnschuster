"""Tests for the app layer (app.py): the controller and its local web API."""
import contextlib
import gc
import hashlib
import http.client
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import app  # noqa: E402
import audio_engine as ae  # noqa: E402
import sacore  # noqa: E402
import updater  # noqa: E402
import syncaudio as sa  # noqa: E402
import vdevice as vd  # noqa: E402
from test_syncaudio import FakeSoundDevice, free_port, hermetic, wait_until  # noqa: E402

# Diagnostics, spec 11: what an output was opened as (11.1) and each second's numbers (11.2).
OPENED = {"engine", "sacore", "sacore_version", "device", "pa", "plan", "observed", "src_active", "call_profile",
          "stream_rate", "opened_reason"}
PLAN = {"blocksize", "latency_s", "level", "bt_class", "fixed_ms", "source", "reason"}
OBSERVED = {"frames_min", "frames_max", "callbacks_per_s", "lead_ms", "probe_d_ms", "props_d_ms", "d_mismatch_ms"}
PLAYER = {"out", "uid", "io", "cb_per_s", "cb_max_us", "skips", "skipped_ms", "overloads", "pa_flags", "underruns",
          "underrun_ms", "ring_lead_min_ms", "ahead_ms", "headroom_min_ms", "spare_1s_ms", "spare_30_ms",
          "deficit_ms", "err_ms", "err_max_ms", "splices", "reseeks", "dll_ppm", "dll_jitter_ms", "dll_reanchors",
          "lossy", "loss_pct", "producer_max_ms", "gc_max_ms", "gil_probe_max_ms", "lead_ms"}
LISTENER = {"arrival_slack_min_ms", "arrival_slack_p1_ms", "holes", "clock_offset_ms", "clock_drift_ppm", "rtt_ms",
            "clock_slew_ms_s", "clock_steps"}
PEER = {"out", "uid", "queue", "unsent", "late", "send_stall_max_ms", "skips", "overloads", "err_max_ms", "gc_max_ms"}
HOST = {"delay_ms", "delay_target_ms", "glide", "cap_io", "cap_skips", "cap_overflows", "cap_gap_frames",
        "cap_reader_max_ms", "cap_dll_ppm"}


class AppTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.opened = []
        self.port = free_port()
        self.servers = []
        self.hermetic = hermetic()          # this computer's real sound output is never touched
        self.hermetic.__enter__()

    def tearDown(self):
        self.hermetic.close()
        for controller, server in self.servers:
            controller.shutdown()
            server.shutdown()
            server.server_close()

    def make(self, name, sd=None):
        c = app.Controller(sd=sd or FakeSoundDevice(), prefs_file=self.tmp / f"{name}.json", port=self.port,
                           scan=False, opener=self.opened.append)
        c.name = name
        server = app.make_server(c)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.servers.append((c, server))
        page = self.request(server, "GET", "/")[1]
        return c, server, page.split('data-token="')[1].split('"')[0]

    def request(self, server, method, path, body=None, token=None, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        headers = {"Host": host or f"127.0.0.1:{server.server_port}"}
        if token:
            headers["X-Token"] = token
        conn.request(method, path, json.dumps(body) if body is not None else None, headers)
        res = conn.getresponse()
        data = res.read().decode()
        conn.close()
        return res.status, data

    def api(self, server, token, action=None, body=None):
        status, data = (self.request(server, "POST", f"/api/{action}", body or {}, token) if action
                        else self.request(server, "GET", "/api/state", token=token))
        self.assertEqual(status, 200, data)
        return json.loads(data)

    def test_api_only_answers_this_apps_window(self):
        c, server, token = self.make("Den")
        self.assertEqual(self.request(server, "GET", "/api/state")[0], 403)                  # no token
        self.assertEqual(self.request(server, "GET", "/api/state", token="nope")[0], 403)
        self.assertEqual(self.request(server, "POST", "/api/stop", {}, token=token, host="evil.example:80")[0], 403)
        self.assertEqual(self.request(server, "GET", "/", host="evil.example")[0], 403)      # DNS rebinding
        state = self.api(server, token)
        self.assertEqual((state["mode"], state["name"], state["blackhole"]), ("idle", "Den", True))

    def test_share_and_join_through_the_api(self):
        _, host_srv, host_token = self.make("Den")
        listener, join_srv, join_token = self.make("Sam")
        self.assertEqual(self.api(host_srv, host_token, "share")["mode"], "hosting")
        self.assertEqual(self.api(join_srv, join_token, "join", {"address": "127.0.0.1"})["mode"], "listening")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            mine = self.api(join_srv, join_token)["session"]
            theirs = self.api(host_srv, host_token)["session"]["listeners"]
            if mine["state"] == "in_sync" and [l["state"] for l in theirs] == ["in_sync"]:
                break
            time.sleep(0.25)
        self.assertEqual((mine["host_name"], mine["state"]), ("Den", "in_sync"))
        self.assertEqual([(l["name"], l["state"]) for l in theirs], [("Sam", "in_sync")])

        state = self.api(join_srv, join_token, "settings", {"volume": 0.4, "offset_ms": -8, "channel": "left"})
        self.assertEqual((state["settings"]["channel"], state["settings"]["offset_ms"]), ("left", -8.0))
        saved = json.loads((self.tmp / "Sam.json").read_text())
        self.assertEqual((saved["volume"], saved["channel"], saved["last_host"]), (0.4, "left", "127.0.0.1"))
        # the nudge belongs to the output it was made on (its UID; its name without the macOS helper)
        self.assertEqual((saved["devices"]["MacBook Pro Speakers"]["offset_ms"], saved["offset_ms"]), (-8.0, 0.0))
        self.assertEqual(self.api(join_srv, join_token, "stop")["mode"], "idle")

    def test_diagnostics_have_every_field_while_sharing_and_listening(self):
        gc.collect()
        frozen = gc.get_freeze_count()      # (not 0 on Python 3.12: a full collection puts immortal objects there)
        host, host_srv, host_token = self.make("Den")
        sam, join_srv, join_token = self.make("Sam")
        self.api(host_srv, host_token, "share")
        self.api(join_srv, join_token, "join", {"address": "127.0.0.1"})
        self.assertTrue(wait_until(lambda: self.api(join_srv, join_token)["session"]["state"] == "in_sync" and
                                   self.api(host_srv, host_token)["session"]["listeners"][0].get("gc_max_ms")
                                   is not None, 20))
        self.api(join_srv, join_token, "settings", {"offset_ms": 3})
        time.sleep(1.2)                     # a second more of history
        self.assertGreater(gc.get_freeze_count(), frozen, "what existed when the music started is never walked again")
        hosting, listening = json.loads(host.diagnostics()), json.loads(sam.diagnostics())
        for report in (hosting, listening):
            self.assertLessEqual({"app", "system", "network", "devices", "audio", "session", "delay_events",
                                  "clock_pings", "python", "selftest", "history", "events", "coreaudiod_overloads"},
                                 set(report))
            out = report["audio"]["output"]
            self.assertLessEqual(OPENED, set(out))
            self.assertLessEqual(PLAN, set(out["plan"]))
            self.assertLessEqual(OBSERVED, set(out["observed"]))
            self.assertEqual((out["plan"]["blocksize"], out["observed"]["frames_max"]), (1024, 1024))
            self.assertIn("gil-probe", report["python"]["threads"])
            self.assertEqual(report["python"]["switch_interval_ms"], round(sys.getswitchinterval() * 1000, 3))
        self.assertEqual(hosting["audio"]["capture"]["plan"]["blocksize"], 512)
        sample = hosting["history"][-1]
        self.assertLessEqual(HOST, set(sample))
        self.assertLessEqual(PLAYER, set(sample["local"]))
        self.assertLessEqual(PEER, set(sample["listeners"][0]))
        self.assertLessEqual(PLAYER | LISTENER, set(listening["history"][-1]))
        self.assertTrue(listening["clock_pings"])
        self.assertIsNone(hosting["clock_pings"])
        events = listening["session"]["events"]
        self.assertEqual([e["event"] for e in events][:1], ["output_open"])
        self.assertIn({"event": "offset_change", "uid": "MacBook Pro Speakers", "ms": 3.0},
                      [{k: v for k, v in e.items() if k != "t"} for e in events])

        text = host.summary(hosting)
        self.assertRegex(text, r"(?m)^this Mac: out MacBook Pro Speakers io 1024 lead [\d.]+ ms \| "
                               r"skips \d+ \([\d.]+ ms\) overloads 0 underruns 0 \| err max [\d.]+ ms \| "
                               r"spare30 [\d.]+ ms \| dll -?[\d.]+ ppm jit [\d.]+ ms \| "
                               r"delay \d+ ms \(\d+ steps, \d+ glides\)\n"
                               r"  static check: \d+ skipped cycles / 0 overloads in the last \d+ s$")
        self.assertRegex(text, r"(?m)^capture BlackHole 2ch io 512 skips \d+ ")
        self.assertRegex(text, r"(?m)^listener Sam \(v2\.0\.0, \w+, wifi [\d.]+ ms, late \d+, unsent \d+ B\): "
                               r"out MacBook Pro Speakers io 1024 ")
        self.assertRegex(sam.summary(listening), r"(?m)^listening to Den \(v2\.0\.0, connected, \w+, wifi [\d.]+ ms, "
                                                 r"silent [\d.]+ s\): out MacBook Pro Speakers io 1024 .*\n"
                                                 r"  static check: ")

        self.api(join_srv, join_token, "stop")
        self.api(host_srv, host_token, "stop")
        self.assertLessEqual(gc.get_freeze_count(), frozen, "a stopped session can be collected")
        saved = json.loads((self.tmp / "Den.json").read_text())       # what the outputs taught us, kept
        self.assertGreater(saved["devices"]["MacBook Pro Speakers"]["lead_ms"], 0)
        self.assertGreater(saved["peer_leads"]["Sam"]["lead_ms"], 0)

    def test_a_friend_on_airpods_last_week_is_covered_from_the_start(self):
        week, month = time.time() - 7 * 86400, time.time() - 31 * 86400
        for seen, delay_ms in ((week, 590), (month, 450)):     # (590: room for AirPods' ~0.2 s to the ear)
            (self.tmp / "Den.json").write_text(json.dumps({"peer_leads": {"Saifan": {"lead_ms": 203.0, "seen": seen}}}))
            _, server, token = self.make("Den")
            self.assertEqual(self.api(server, token, "share")["session"]["delay_ms"], delay_ms)
            self.api(server, token, "stop")

    def test_headphone_test_through_the_api(self):
        sd = vd.FakeSD([{"name": "AirPods", "max_output_channels": 2, "default_low_output_latency": 0.165,
                         "lead": 0.16, "frames": 64}])
        sd.default = types.SimpleNamespace(device=[None, 0])       # the Mac's output: the AirPods
        c, server, token = self.make("Den", sd)
        with mock.patch.object(ae.SelfTest, "SECONDS", (0.6, 2.5)):
            test = self.api(server, token, "selftest")["selftest"]
            self.assertEqual((test["running"], test["output"], test["seconds"]), (True, "AirPods", [0.6, 2.5]))
            self.assertEqual(self.api(server, token, "share")["error"]["code"], "busy", "it has the output to itself")
            self.assertTrue(wait_until(lambda: not self.api(server, token)["selftest"]["running"], 20))
        test = self.api(server, token)["selftest"]
        r = test["result"]
        self.assertEqual((r["legacy"]["frames_max"], r["v2"]["io_frames"]), (64, 2048))   # 1.x's buffer, then 2.0's
        self.assertEqual(test["message"], sa.VERDICTS[r["verdict"]])
        report = json.loads(c.diagnostics())
        self.assertEqual(report["selftest"]["result"]["verdict"], r["verdict"])
        self.assertIn(f"headphone test on AirPods: {r['verdict']} | old way io ", c.summary(report))

    def test_headphone_test_waits_until_the_session_stops(self):
        _, server, token = self.make("Den")
        self.api(server, token, "join", {"address": "127.0.0.1"})     # (nobody shares there: still a session)
        state = self.api(server, token, "selftest")
        self.assertEqual((state["error"]["code"], state["selftest"]["running"]), ("busy", False))

    def test_the_rooms_choice_waits_for_the_headphone_test(self):
        # the other Mac picks this one's music while its headphone test runs: shared after the test,
        # not undone for everyone
        c, _, _ = self.make("Den")
        calls = []
        c.room = types.SimpleNamespace(source={"id": c.device_id, "at": 1.0}, members={}, stop=lambda: None,
                                       set_source=calls.append)
        c.selftest = {"running": True}
        c._apply_room()
        self.assertEqual((calls, c.session), ([], None))
        c.selftest = {"running": False}
        c._apply_room()
        self.assertIsInstance(c.session, sa.HostSession)
        c.room = None

    def test_listening_the_room_asks_for_waits_for_the_headphone_test_too(self):
        # with no red "the test is still running" for something this person didn't ask for
        c, _, _ = self.make("Den")
        c.room = types.SimpleNamespace(source={"id": "f", "at": 1.0}, members={"f": {"ip": "127.0.0.1", "port": 1}},
                                       stop=lambda: None, set_source=lambda sid: None)
        c.selftest = {"running": True}
        c._apply_room()
        self.assertEqual((c.error, c.session), (None, None))
        c.selftest = {"running": False}
        c._apply_room()
        self.assertIsInstance(c.session, sa.JoinSession)
        c.room = None

    def test_joining_asks_portaudio_for_the_devices_again(self):
        # AirPods put in since the app opened: PortAudio only knows them (and that they are the Mac's
        # output now) once restarted, as before sharing
        c, _, _ = self.make("Sam")
        with mock.patch.object(sa, "refresh_portaudio") as refresh:
            self.assertTrue(c.join("127.0.0.1"))
        refresh.assert_called_once_with(c.sd)

    def test_sharing_without_blackhole_explains_the_setup(self):
        sd = FakeSoundDevice()
        sd.DEVICES = [d for d in sd.DEVICES if "BlackHole" not in d["name"]]
        _, server, token = self.make("Den", sd)
        state = self.api(server, token, "share")
        self.assertEqual((state["mode"], state["blackhole"], state["error"]["code"]), ("idle", False, "no_blackhole"))

    def test_only_known_links_open(self):
        _, server, token = self.make("Den")
        self.api(server, token, "open", {"what": "file:///etc/passwd"})
        self.api(server, token, "open", {"what": "microphone"})
        self.assertEqual(self.opened, [app.LINKS["microphone"]])

    def test_installs_blackhole_and_notices_it(self):
        sd = FakeSoundDevice()
        blackhole = sd.DEVICES[0]
        sd.DEVICES = sd.DEVICES[1:]
        c, server, token = self.make("Den", sd)
        self.assertFalse(self.api(server, token)["blackhole"])
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd[0])
            if cmd[0] == "curl":
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"pkg")
            if cmd[0] == "osascript":
                self.assertIn("with administrator privileges", cmd[2])
                sd.DEVICES = [blackhole] + sd.DEVICES        # the driver appears
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch.object(app.subprocess, "run", fake_run), \
                mock.patch.object(app, "BLACKHOLE_SHA256", hashlib.sha256(b"pkg").hexdigest()), \
                mock.patch.object(app.time, "sleep", lambda s: None):
            c._install_blackhole()
        self.assertEqual(calls, ["curl", "osascript"])
        state = self.api(server, token)
        self.assertEqual((state["install"]["state"], state["blackhole"]), ("done", True))
        self.assertEqual(self.api(server, token, "share")["mode"], "hosting")

    def test_blackhole_install_cancel_and_bad_download(self):
        sd = FakeSoundDevice()
        sd.DEVICES = sd.DEVICES[1:]
        c, _, _ = self.make("Den", sd)

        def cancelled(cmd, **kw):
            if cmd[0] == "curl":
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"pkg")
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return subprocess.CompletedProcess(cmd, 1, "", "execution error: User canceled. (-128)")

        with mock.patch.object(app.subprocess, "run", cancelled), \
                mock.patch.object(app, "BLACKHOLE_SHA256", hashlib.sha256(b"pkg").hexdigest()):
            c._install_blackhole()
        self.assertEqual(c.install["state"], "idle")
        with mock.patch.object(app.subprocess, "run", cancelled):     # hash won't match
            c._install_blackhole()
        self.assertEqual(c.install["state"], "failed")
        self.assertIn("didn't match", c.install["message"])

    def test_puts_the_sound_output_back_after_a_crash(self):
        prefs = self.tmp / "Den.json"
        prefs.write_text(json.dumps({"restore_output": "MacBook Pro Speakers"}))
        switched = []
        with mock.patch.object(sa.SystemOutput, "current", return_value="BlackHole 2ch"), \
                mock.patch.object(sa.SystemOutput, "set", lambda self, name: switched.append(name)):
            app.Controller(sd=FakeSoundDevice(), prefs_file=prefs, scan=False)
        self.assertEqual(switched, ["MacBook Pro Speakers"])
        self.assertNotIn("restore_output", json.loads(prefs.read_text()))


if __name__ == "__main__":
    unittest.main()


class UpdaterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def bundle(self, name, version):
        app_dir = self.tmp / name
        (app_dir / "Contents").mkdir(parents=True)
        (app_dir / "Contents" / "version").write_text(version)
        return app_dir

    def test_swap_replaces_the_app_after_it_quits_and_reopens_it(self):
        old = self.bundle("SyncAudio.app", "1")
        staging = self.tmp / ".SyncAudio-update-x"
        staging.mkdir()
        new = staging / "SyncAudio.app"
        self.bundle(".SyncAudio-update-x/SyncAudio.app", "2")
        running = subprocess.Popen(["sleep", "0.5"])         # stands in for the app that's quitting
        threading.Thread(target=running.wait, daemon=True).start()   # reap it like launchd would
        marker = self.tmp / "reopened"
        script = updater.swap_script(old, new, staging, running.pid, f"touch {marker}")
        start = time.monotonic()
        subprocess.run(["/bin/sh", "-c", script], check=True, timeout=30)
        self.assertGreaterEqual(time.monotonic() - start, 0.3, "waited for the app to quit")
        self.assertEqual((old / "Contents" / "version").read_text(), "2")
        self.assertTrue(marker.exists())
        self.assertFalse(staging.exists(), "cleaned up")

    def test_a_failed_swap_keeps_the_old_app(self):
        old = self.bundle("SyncAudio.app", "1")
        staging = self.tmp / ".SyncAudio-update-y"
        staging.mkdir()
        script = updater.swap_script(old, staging / "missing.app", staging, 999999, None)
        subprocess.run(["/bin/sh", "-c", script], timeout=30)
        self.assertEqual((old / "Contents" / "version").read_text(), "1")

    def test_reads_the_latest_release(self):
        with mock.patch.object(updater, "_run", return_value=json.dumps({"version": "1.0.9", "commit": "abc123"})):
            self.assertEqual(updater.latest_release(),
                             {"version": "1.0.9", "commit": "abc123", "dmg_url": updater.DMG_URL})
        with mock.patch.object(updater, "_run", side_effect=RuntimeError("offline")):
            self.assertIsNone(updater.latest_release())

    def test_offers_an_update_only_when_the_build_differs(self):
        c = app.Controller(sd=FakeSoundDevice(), prefs_file=self.tmp / "p.json", scan=False)
        latest = {"version": "1.0.9", "commit": "new", "dmg_url": "https://example.com/SyncAudio.dmg"}
        with mock.patch.object(updater, "latest_release", return_value=latest):
            c.build = {"version": "1.0.8", "commit": "old"}
            c.check_update()
            self.assertTrue(c.state()["update"]["available"])
            self.assertNotIn("dmg_url", c.state()["update"])
            c.build = {"version": "1.0.9", "commit": "new"}
            c.check_update()
            self.assertFalse(c.state()["update"]["available"])
            c.build = {"version": "1.0.0", "commit": None}     # running from source
            c.check_update()
            self.assertFalse(c.state()["update"]["available"])

    def test_source_checkout_cannot_replace_itself(self):
        c = app.Controller(sd=FakeSoundDevice(), prefs_file=self.tmp / "p.json", scan=False)
        c.update_info.update(dmg_url="https://example.com/SyncAudio.dmg")
        c._install_update()
        self.assertEqual(c.update_info["state"], "failed")


class SummaryTest(unittest.TestCase):
    def test_static_check_counts_the_last_minute_and_survives_a_reopened_output(self):
        p = {"out": "AirPods", "io": [1024, 2048], "lead_ms": 203.1, "skips": 9, "skipped_ms": 383.9, "overloads": 2,
             "underruns": 0, "err_max_ms": 0.3, "spare_30_ms": 180.2, "dll_ppm": 7400.5, "dll_jitter_ms": 0.21,
             "lossy": True, "loss_pct": 0.74}
        # 90 s of history: skips 5 -> 7 -> 9, the last two steps in the last minute; overloads 6, then the
        # output was reopened (its counter starts again), then 2
        past = [(t, {"skips": 5 if t < 30 else 7 if t <= 60 else 9, "overloads": 6 if t < 50 else 0 if t < 70 else 2,
                     "err_max_ms": 2.5 if t == 10 else 0.1}) for t in range(90)]
        self.assertEqual(app.player_summary("this Mac", p, past, "delay 590 ms (1 steps, 0 glides)"), [
            "this Mac: out AirPods io 1024-2048 lead 203.1 ms | skips 9 (383.9 ms) overloads 2 underruns 0 | "
            "err max 2.5 ms | spare30 180.2 ms | dll 7400.5 ppm jit 0.21 ms LOSSY 0.74 % | "
            "delay 590 ms (1 steps, 0 glides)",
            "  static check: 4 skipped cycles / 2 overloads in the last 60 s"])
        self.assertEqual(app.player_summary("x", p, [], "d")[1:], [], "no history yet: no static check")

    def test_headphone_test_line(self):
        r = {"output": "AirPods", "verdict": "macos_dropping",
             "legacy": {"frames_min": 15, "frames_max": 64, "lost_ms_per_s": 7.33},
             "v2": {"io_frames": 2048, "skips": 3, "skipped_ms": 128.0, "overloads": 3, "underruns": 0,
                    "cb_max_us": 41.0, "lossy": False}}
        self.assertEqual(app.selftest_summary(r), "headphone test on AirPods: macos_dropping | old way io 15-64 lost "
                                                  "7.33 ms/s | 2.0 io 2048 skips 3 (128.0 ms) overloads 3 underruns 0 "
                                                  "cb max 41.0 us")

    def test_system_facts(self):
        answers = {("/usr/sbin/sysctl", "-n", "hw.model"): "Mac15,6",
                   ("/usr/bin/pmset", "-g"): "System-wide power settings:\n lowpowermode         1\n sleep 1",
                   ("/usr/bin/pmset", "-g", "therm"): "No thermal warning level has been recorded"}
        self.assertEqual(app.system_info(lambda *cmd: answers[cmd]),
                         {"model": "Mac15,6", "low_power_mode": "1",
                          "thermal": "No thermal warning level has been recorded"})


class PrefsTest(unittest.TestCase):
    def test_each_output_keeps_its_nudge_by_uid(self):
        prefs = Path(tempfile.mkdtemp()) / "p.json"
        airpods = {"name": "AirPods", "uid": "AirPods-UID", "offset_ms": -12.0, "io_level": 1}
        prefs.write_text(json.dumps({"offset_ms": 4.0, "devices": {"AirPods-UID": airpods}}))
        c = app.Controller(sd=FakeSoundDevice(), prefs_file=prefs, scan=False)
        self.assertEqual(c.settings.offset_ms, 4.0)         # set with no output open: for those without their own
        c.settings.device_key = "AirPods-UID"               # what an opened output does
        self.assertEqual(c.settings.offset_ms, -12.0)
        c.update({"offset_ms": -10})
        saved = json.loads(prefs.read_text())
        self.assertEqual((saved["devices"]["AirPods-UID"], saved["offset_ms"]), ({**airpods, "offset_ms": -10.0}, 4.0))


    def test_a_1x_nudge_starts_again_at_0(self):
        # it made up for 1.x's own timing, which drifted late: carried over, every 2.0 output would be off
        prefs = Path(tempfile.mkdtemp()) / "p.json"
        prefs.write_text(json.dumps({"volume": 1.0, "offset_ms": 21.0, "channel": "both", "name": "Farhan"}))
        c = app.Controller(sd=FakeSoundDevice(), prefs_file=prefs, scan=False)
        c.settings.device_key = "Farhan’s AirPods Pro"
        self.assertEqual(c.settings.offset_ms, 0.0)
        self.assertEqual(json.loads(c.diagnostics())["settings"]["offset_ms_1x"], 21.0)
        c.update({"volume": 0.5})
        saved = json.loads(prefs.read_text())
        self.assertEqual((saved["offset_ms"], saved["offset_ms_1x"], "devices" in saved), (0.0, 21.0, True))
        c2 = app.Controller(sd=FakeSoundDevice(), prefs_file=prefs, scan=False)       # and only the first time
        self.assertEqual((c2.settings.offset_ms, c2.prefs["offset_ms_1x"]), (0.0, 21.0))


class BuildSelfTestTest(unittest.TestCase):
    """--self-test, the build's check of the packaged app, with stand-ins for sounddevice and pywebview."""

    def run_self_test(self, **changes):
        fake = FakeSoundDevice()
        sd = types.SimpleNamespace(__version__="0.5.6", _StreamBase=fake._StreamBase, query_devices=fake.query_devices,
                                   get_portaudio_version=fake.get_portaudio_version, default=fake.default)
        for k, v in changes.items():
            setattr(sd, k, v)
        out = io.StringIO()
        with mock.patch.dict(sys.modules, {"sounddevice": sd, "webview": types.ModuleType("webview")}), hermetic(), \
                contextlib.redirect_stdout(out):
            code = app.self_test()
        return code, out.getvalue()

    @unittest.skipUnless(sacore.available(), "libsacore isn't built")
    def test_checks_the_audio_core(self):
        code, out = self.run_self_test()
        self.assertEqual(code, 0, out)
        self.assertIn(f"sacore: {sacore.LIBRARY} 0x20000", out)
        self.assertEqual(self.run_self_test(__version__="0.5.5")[0], 1)
        with mock.patch.object(sacore, "now", lambda: time.monotonic() + 0.01):
            self.assertIn("clock", self.run_self_test()[1])
        with mock.patch.object(sacore, "smoke_test", return_value=False):
            self.assertIn("didn't play back", self.run_self_test()[1])

    @unittest.skipUnless(sacore.available(), "libsacore isn't built")
    def test_the_app_must_use_its_own_copy(self):
        inside = tempfile.mkdtemp()             # the app's Contents/Frameworks: the page is in there too
        os.symlink(os.path.join(os.path.dirname(app.__file__), "ui"), os.path.join(inside, "ui"))
        with mock.patch.object(sys, "frozen", True, create=True), \
                mock.patch.object(sys, "_MEIPASS", inside, create=True), \
                mock.patch.object(sa.codec, "available", return_value=True):
            self.assertIn("not from inside the app", self.run_self_test()[1])
            with mock.patch.object(sacore, "lib", None):
                self.assertIn("isn't in the app", self.run_self_test()[1])
            with mock.patch.object(sacore, "LIBRARY", os.path.join(inside, "libsacore.dylib")):
                self.assertEqual(self.run_self_test()[0], 0)
        with mock.patch.object(sacore, "lib", None):       # running from source without a build: says so, no fail
            self.assertEqual(self.run_self_test()[0], 0)


class FirewallTest(unittest.TestCase):
    def test_reads_the_firewall(self):
        on, off = "Firewall is enabled. (State = 1)", "Firewall is disabled. (State = 0)"
        self.assertFalse(app.parse_firewall(off, "", "")["needs_fix"])
        self.assertTrue(app.parse_firewall(on, "", "The application is not part of the firewall")["needs_fix"])
        blocked = app.parse_firewall(on, "", "Incoming connection to the application is blocked")
        self.assertEqual((blocked["app"], blocked["needs_fix"]), ("blocked", True))
        allowed = app.parse_firewall(on, "", "Incoming connection to the application is permitted")
        self.assertEqual((allowed["app"], allowed["needs_fix"]), ("allowed", False))
        self.assertTrue(app.parse_firewall(on, "Firewall is set to block all non-essential incoming connections",
                                           "Incoming connection to the application is permitted")["needs_fix"])

    def test_asked_when_sharing_starts_then_kept_five_minutes(self):
        c = app.Controller(sd=FakeSoundDevice(), prefs_file=Path(tempfile.mkdtemp()) / "p.json", port=free_port(),
                           scan=False)
        asked = []
        with mock.patch.object(app.sys, "platform", "darwin"), \
                mock.patch.object(app.os.path, "exists", return_value=True), \
                mock.patch.object(app, "firewall_status", lambda path: asked.append(path) or {"needs_fix": False}):
            c.firewall()
            c.firewall()
            c._firewall_at -= 299
            c.firewall()
            self.assertEqual(len(asked), 1, "three processes each time: not every few seconds")
            c._firewall_at -= 2
            c.firewall()
            c.firewall(fresh=True)
            self.assertEqual(len(asked), 3)
        with mock.patch.object(c, "firewall") as firewall, hermetic():
            self.assertTrue(c.share())
            firewall.assert_called_once_with(fresh=True)
            c.shutdown()

    def test_fix_asks_for_the_password_once(self):
        c = app.Controller(sd=FakeSoundDevice(), prefs_file=Path(tempfile.mkdtemp()) / "p.json", scan=False)
        status = {**app.parse_firewall("Firewall is enabled. (State = 1)", "Firewall is set to block all",
                                       "is blocked"), "app_path": "/Applications/Sync Audio's.app"}
        calls = []
        with mock.patch.object(c, "firewall", return_value=status), \
                mock.patch.object(app.subprocess, "run",
                                  lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", "")):
            c.fix_firewall()
        self.assertEqual(len(calls), 1)
        script = " ".join(calls[0])
        self.assertIn("with administrator privileges", script)
        for part in ("--add", "--unblockapp", "--setblockall off", "set p to \"/Applications/Sync Audio's.app\""):
            self.assertIn(part, script)
