"""Tests for finding sharing Macs (discovery.py), plus a real Bonjour round trip on macOS."""
import ipaddress
import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import app  # noqa: E402
import discovery  # noqa: E402
import syncaudio as sa  # noqa: E402
from test_syncaudio import FakeSoundDevice, free_port, wait_until  # noqa: E402

IFCONFIG = """en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether 3c:22:fb:00:00:01
\tinet6 fe80::1c2b:aaaa:bbbb:cccc%en0 prefixlen 64 secured scopeid 0xb
\tinet 10.48.71.96 netmask 0xfffff000 broadcast 10.48.79.255
\tstatus: active
lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
\tinet 127.0.0.1 netmask 0xff000000
"""


class SubnetTest(unittest.TestCase):
    def test_reads_the_real_netmask(self):
        net, broadcast = discovery.interface_for("10.48.71.96", IFCONFIG)
        self.assertEqual((str(net), broadcast), ("10.48.64.0/20", "10.48.79.255"))
        self.assertIsNone(discovery.interface_for("192.168.1.5", IFCONFIG))

    def test_sweeps_the_whole_subnet_nearest_first(self):
        sweep = discovery.Sweep("10.48.71.96", ipaddress.IPv4Network("10.48.64.0/20"), chunk=512)
        first = sweep.next_targets()
        self.assertEqual(first[:2], ["10.48.71.95", "10.48.71.97"])
        self.assertNotIn("10.48.71.96", first)
        seen = set(first)
        for _ in range(7):
            seen.update(sweep.next_targets())
        self.assertEqual(len(seen), 4093, "every other address on the /20 in 8 scans")

    def test_huge_networks_are_limited_to_the_surrounding_slash_15(self):
        sweep = discovery.Sweep("10.48.71.96", ipaddress.IPv4Network("10.0.0.0/8"))
        self.assertEqual(len(sweep.order), 131069)

    def test_a_campus_slash_15_is_swept_whole(self):
        # Farhan (10.48.71.96) and Saifan (10.49.64.113) share 10.48.0.0/15.
        sweep = discovery.Sweep("10.48.71.96", ipaddress.IPv4Network("10.48.0.0/15"))
        self.assertIn("10.49.64.113", sweep.order)


class BonjourParsingTest(unittest.TestCase):
    def test_browse_lines(self):
        line = "13:45:01.123  Add        3   5 local.               _syncaudio._tcp.     Farhan’s MacBook Pro\n"
        self.assertEqual(discovery.parse_browse_line(line), ("Add", "Farhan’s MacBook Pro"))
        self.assertEqual(discovery.parse_browse_line(line.replace("Add", "Rmv")), ("Rmv", "Farhan’s MacBook Pro"))
        self.assertIsNone(discovery.parse_browse_line("Browsing for _syncaudio._tcp.local\n"))

    def test_resolve_output(self):
        text = ("13:45:01.456  Farhan’s\\032MacBook\\032Pro._syncaudio._tcp.local. can be reached at "
                "Farhans-MacBook-Pro.local.:47800 (interface 5) Flags: 1\n ip=10.48.71.96 v=1\n")
        self.assertEqual(discovery.parse_resolve(text), ("10.48.71.96", 47800))


class RememberedMacsTest(unittest.TestCase):
    def test_a_mac_played_with_before_is_found_directly(self):
        port = free_port()
        host = sa.Host(48000, 2, 0.1, port, "Den", log=lambda *a: None)
        host.start()
        prefs = Path(tempfile.mkdtemp()) / "prefs.json"
        prefs.write_text(json.dumps({"known_hosts": ["127.0.0.1"]}))
        c = app.Controller(sd=FakeSoundDevice(), prefs_file=prefs, port=port)
        try:
            self.assertTrue(wait_until(lambda: [h["name"] for h in c.state()["nearby"]] == ["Den"], 10),
                            c.state()["nearby"])
        finally:
            c.shutdown()
            host.stop()


@unittest.skipUnless(discovery.dns_sd(), "needs macOS's dns-sd")
class BonjourRoundTripTest(unittest.TestCase):
    def test_a_sharing_mac_announces_its_address(self):
        name = f"SyncAudio test {uuid.uuid4().hex[:6]}"
        ad = discovery.Advertiser(name, 47999, sa.lan_ip()).start()
        browser = discovery.Browser().start()
        try:
            self.assertTrue(wait_until(lambda: browser.found.get(name) == sa.lan_ip(), 20), browser.found)
            ad.stop()
            self.assertTrue(wait_until(lambda: name not in browser.found, 20), "gone once it stops")
        finally:
            ad.stop()
            browser.stop()


if __name__ == "__main__":
    unittest.main()


class ProbeTest(unittest.TestCase):
    def test_probe_checks_a_sharing_mac_end_to_end(self):
        port = free_port()
        host = sa.Host(48000, 2, 0.1, port, "Den", log=lambda *a: None)
        host.start()
        clicks = sa.ClickSource(48000, 2, host.on_block)
        clicks.start()
        try:
            result = sa.probe("127.0.0.1", port, seconds=1.0)
            self.assertTrue(result["ok"], result)
            self.assertEqual((result["host_name"], result["discovery"]), ("Den", True))
            self.assertGreater(result["audio_blocks_per_s"], 75)           # 10 ms blocks
            self.assertEqual(host.listeners(), [], "a connection test isn't shown as a listener")
        finally:
            clicks.close()
            host.stop()

    def test_probe_explains_failures(self):
        result = sa.probe("127.0.0.1", free_port(), seconds=0.5)
        self.assertEqual((result["ok"], result["step"], result["problem"]["code"]), (False, "connect", "refused"))

    def test_errors_in_plain_words(self):
        import errno
        import socket
        self.assertEqual(sa.explain(socket.timeout())["code"], "timeout")
        self.assertEqual(sa.explain(OSError(errno.EHOSTUNREACH, "No route to host"))["code"], "local_network")
        self.assertEqual(sa.explain(OSError(errno.ECONNREFUSED, "refused"))["code"], "refused")
        self.assertIn("Local Network", sa.explain(OSError(errno.EHOSTUNREACH, "x"))["message"])


class QuitTest(unittest.TestCase):
    def test_quitting_stops_the_background_search(self):
        stopped = []

        class FakeBrowser:
            found = {}

            def start(self):
                return self

            def stop(self):
                stopped.append(True)

        from unittest import mock
        with mock.patch.object(discovery, "Browser", FakeBrowser):
            c = app.Controller(sd=FakeSoundDevice(), prefs_file=Path(tempfile.mkdtemp()) / "p.json", port=free_port())
            self.assertTrue(wait_until(lambda: c.browser is not None, 5))
            c.shutdown()
        self.assertEqual(stopped, [True], "dns-sd must not outlive the app")


class DiagnosticsTest(unittest.TestCase):
    def test_report_has_what_we_need(self):
        c = app.Controller(sd=FakeSoundDevice(), prefs_file=Path(tempfile.mkdtemp()) / "p.json", scan=False)
        report = json.loads(c.diagnostics())
        for key in ("app", "macos", "network", "devices", "session", "history", "events"):
            self.assertIn(key, report)
        self.assertEqual(report["devices"][0]["name"], "BlackHole 2ch")

    def test_copy_gives_a_whole_summary_and_saves_the_full_report(self):
        # Copy diagnostics used to cut everything off after 500 characters.
        home = Path(tempfile.mkdtemp())
        (home / "Desktop").mkdir()
        c = app.Controller(sd=FakeSoundDevice(), prefs_file=home / "p.json", scan=False)
        copied = []
        with mock.patch.object(app.Path, "home", return_value=home), \
                mock.patch.object(app.subprocess, "run", wraps=app.subprocess.run) as run:
            run.side_effect = lambda cmd, **kw: copied.append(kw["input"]) if cmd == ["pbcopy"] else \
                mock.DEFAULT
            c.copy_diagnostics()
        text = copied[-1]
        self.assertIn("SyncAudio " + app.VERSION, text)
        self.assertTrue(text.rstrip().endswith("SyncAudio diagnostics.txt"), text)
        full = json.loads((home / "Desktop" / "SyncAudio diagnostics.txt").read_text())
        self.assertIn("history", full)
