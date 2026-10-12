"""Tests for the app layer (app.py): the controller and its local web API."""
import hashlib
import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import app  # noqa: E402
import updater  # noqa: E402
import syncaudio as sa  # noqa: E402
from test_syncaudio import FakeSoundDevice, free_port  # noqa: E402


class AppTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.opened = []
        self.port = free_port()
        self.servers = []
        self.which = mock.patch.object(sa.shutil, "which", return_value=None)
        self.which.start()

    def tearDown(self):
        self.which.stop()
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
        self.assertEqual(state["settings"]["channel"], "left")
        saved = json.loads((self.tmp / "Sam.json").read_text())
        self.assertEqual((saved["volume"], saved["offset_ms"], saved["channel"], saved["last_host"]),
                         (0.4, -8.0, "left", "127.0.0.1"))
        self.assertEqual(self.api(join_srv, join_token, "stop")["mode"], "idle")

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
        release = {"name": "SyncAudio 1.0.9", "target_commitish": "abc123",
                   "assets": [{"name": "SyncAudio.dmg", "browser_download_url": "https://example.com/SyncAudio.dmg"}]}
        with mock.patch.object(updater, "_run", return_value=json.dumps(release)):
            self.assertEqual(updater.latest_release(),
                             {"version": "1.0.9", "commit": "abc123", "dmg_url": "https://example.com/SyncAudio.dmg"})
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
