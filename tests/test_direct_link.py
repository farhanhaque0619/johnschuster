"""Wi-Fi that keeps devices apart (client isolation): many school, office and guest networks show
Macs to each other (Bonjour gets through) but drop every connection between them. Then the listener
reaches the sharing Mac over the Macs' own peer-to-peer Wi-Fi link, the one AirDrop uses: IPv6 on
awdl0, found by the host's Bonjour name. That link only exists between real Macs; here the host's
IPv6 loopback stands in for it."""
import os
import socket
import sys
import time
import unittest
from unittest import mock

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
sys.path.insert(0, here)
import audio_engine as ae  # noqa: E402
import discovery  # noqa: E402
import syncaudio as sa  # noqa: E402
from test_syncaudio import RATE, FakeSoundDevice, free_port, has_fake, hermetic, quiet, wait_until  # noqa: E402


def has_ipv6() -> bool:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


ISOLATED_AT = "192.0.2.1"      # a documentation address: stands for the host's address on such a Wi-Fi


def isolated_network(real=sa.connect):
    """Connections to the host's Wi-Fi address time out, as on a network that keeps devices apart."""
    def connect(ip, port, timeout=3.0):
        if ip == ISOLATED_AT:
            raise socket.timeout("timed out")
        return real(ip, port, timeout)
    return mock.patch.object(sa, "connect", side_effect=connect)


class DnsSdParsingTest(unittest.TestCase):
    RESOLVE = (" 2:48:02.614  Farhans-MacBook-Pro._syncaudio._tcp.local. can be reached at "
               "Farhans-MacBook-Pro.local.:47800 (interface 14) Flags: 1\n ip=172.20.1.73 v=1\n"
               " 2:48:02.620  Farhans-MacBook-Pro._syncaudio._tcp.local. can be reached at "
               "Farhans-MacBook-Pro.local.:47800 (interface 12)\n ip=172.20.1.73 v=1\n")
    ADDR = ("DATE: ---Thu 05 Nov 2026---\n 2:48:03.001  ...STARTING...\n"
            "Timestamp     A/R    Flags if Hostname                       Address                             TTL\n"
            " 2:48:03.010  Add 40000002 12 Farhans-MacBook-Pro.local.     "
            "FE80:0000:0000:0000:1C5A:12FF:FE34:5678%awdl0  120\n"
            " 2:48:03.012  Add 40000002 14 Farhans-MacBook-Pro.local.     "
            "FE80:0000:0000:0000:0AAA:0BBB:0CCC:0DDD%en0  120\n")

    def test_every_interface_a_resolve_reached(self):
        self.assertEqual(discovery.parse_targets(self.RESOLVE),
                         [("Farhans-MacBook-Pro.local.", 47800, 14), ("Farhans-MacBook-Pro.local.", 47800, 12)])
        self.assertEqual(discovery.parse_resolve(self.RESOLVE), ("172.20.1.73", 47800))

    def test_ipv6_addresses_as_python_connects_to_them(self):
        self.assertEqual(discovery.parse_addr6(self.ADDR), ["fe80::1c5a:12ff:fe34:5678%awdl0",
                                                            "fe80::aaa:bbb:ccc:ddd%en0"])

    def test_which_interfaces_are_the_airdrop_link(self):
        self.assertTrue(discovery.is_p2p_interface("awdl0"))
        self.assertTrue(discovery.is_p2p_interface("llw0"))
        self.assertFalse(discovery.is_p2p_interface("en0"))
        self.assertFalse(discovery.is_p2p_interface(None))

    def test_finds_the_host_on_the_link_by_name(self):
        """p2p_hosts: resolve the name, keep the target on the AirDrop link, ask for its IPv6 there."""
        def read(cmd, seconds, done=None):
            if "-L" in cmd:
                return self.RESOLVE
            if "-G" in cmd:
                self.assertEqual(cmd[-1], "Farhans-MacBook-Pro.local.")
                self.assertTrue({"-includep2p", "-includeAWDL"} <= set(cmd))
                return self.ADDR
            raise AssertionError(cmd)
        names = {12: "awdl0", 14: "en0"}
        with mock.patch.object(discovery, "dns_sd", return_value="/usr/bin/dns-sd"), \
                mock.patch.object(discovery, "_read", side_effect=read), \
                mock.patch.object(discovery.socket, "if_indextoname", side_effect=names.__getitem__):
            found = discovery.p2p_hosts(names=["Farhans-MacBook-Pro"])
        self.assertEqual(found, [{"name": "Farhans-MacBook-Pro", "ip": "172.20.1.73",
                                  "address": "fe80::1c5a:12ff:fe34:5678%awdl0", "port": 47800}])

    def test_nothing_without_bonjour(self):
        with mock.patch.object(discovery, "dns_sd", return_value=None):
            self.assertEqual(discovery.p2p_hosts(), [])


class HostWithoutIPv6Test(unittest.TestCase):
    def test_shares_over_ipv4_when_this_mac_has_no_ipv6(self):
        real = socket.socket

        def sockets(family=socket.AF_INET, *a, **k):
            if family == socket.AF_INET6:
                raise OSError(97, "Address family not supported by protocol")
            return real(family, *a, **k)
        host = sa.Host(RATE, 2, 0.45, free_port(), "Den", log=quiet)
        with mock.patch.object(sa.socket, "socket", side_effect=sockets):
            host.start()
        try:
            self.assertEqual((host.tcp6, host.udp6), (None, None))
            with socket.create_connection(("127.0.0.1", host.port), timeout=2):
                pass
        finally:
            host.stop()


@unittest.skipUnless(has_ipv6(), "needs IPv6 (the AirDrop link is IPv6)")
@unittest.skipUnless(has_fake, "needs the test build of libsacore")
class DirectLinkTest(unittest.TestCase):
    def test_a_listener_on_wifi_that_keeps_devices_apart_plays_over_the_direct_link(self):
        port = free_port()
        host = sa.Host(RATE, 2, 0.45, port, "Den", log=quiet)
        host.start()
        self.assertIsNotNone(host.tcp6)
        fsd = FakeSoundDevice(15)
        found = [{"name": "Den", "ip": ISOLATED_AT, "address": "::1", "port": port}]

        def open_output(rate, channels, player, reason="start", level=None):
            return ae.AudioOutput(fsd, 2, rate, player, client.settings, {}, log=quiet, reason=reason, level=level)
        client = sa.Client(ISOLATED_AT, open_output, port, "Sam", log=quiet, host_label="Den")
        clicks = sa.ClickSource(RATE, 2, host.on_block)
        with hermetic(), isolated_network(), \
                mock.patch.object(sa.discovery, "dns_sd", return_value="/usr/bin/dns-sd"), \
                mock.patch.object(sa.discovery, "p2p_hosts", return_value=found) as lookup:
            clicks.start()
            client.start()
            try:
                self.assertTrue(wait_until(lambda: any(l["state"] == "in_sync" for l in host.listeners()), 15),
                                host.listeners())
                listeners = host.listeners()
            finally:
                clicks.close()
                client.stop()
                host.stop()
        self.assertEqual([(l["name"], l["ip"]) for l in listeners], [("Sam", "::1")])
        self.assertEqual((client.route, client._direct), ("direct", "::1"))
        self.assertEqual(lookup.call_args.kwargs["names"], ["Den"])        # looked up by its Bonjour name
        self.assertEqual([e["event"] for e in client.events if e["event"] == "direct_link"], ["direct_link"])
        self.assertIsNone(client.problem)


class IsolatedTest(unittest.TestCase):
    def test_says_plainly_when_the_wifi_keeps_the_macs_apart(self):
        client = sa.Client(ISOLATED_AT, lambda *a, **k: None, free_port(), "Sam", log=quiet, host_label="Den")
        with isolated_network(), mock.patch.object(sa.discovery, "dns_sd", return_value="/usr/bin/dns-sd"), \
                mock.patch.object(sa.discovery, "p2p_hosts", return_value=[]):
            client.start()
            try:
                self.assertTrue(wait_until(lambda: client.problem is not None, 5))
            finally:
                client.stop()
        self.assertEqual(client.problem["code"], "isolated")
        self.assertIn("keeps devices apart", client.problem["message"])
        self.assertIn("where this worked before", client.problem["message"])
        self.assertIn("Den", client.problem["message"])

    def test_a_typed_address_that_times_out_keeps_the_old_explanation(self):
        client = sa.Client(ISOLATED_AT, lambda *a, **k: None, free_port(), "Sam", log=quiet)
        with isolated_network(), mock.patch.object(sa.discovery, "dns_sd", return_value=None):
            client.start()
            try:
                self.assertTrue(wait_until(lambda: client.problem is not None, 5))
            finally:
                client.stop()
        self.assertEqual(client.problem["code"], "timeout")


class AdvertiserTest(unittest.TestCase):
    def test_the_share_is_on_the_direct_link_only_while_it_may_be_needed(self):
        cmds = []
        with mock.patch.object(discovery, "dns_sd", return_value="/usr/bin/dns-sd"), \
                mock.patch.object(discovery.subprocess, "Popen", side_effect=lambda cmd, **k: cmds.append(cmd)
                                  or mock.Mock()):
            ad = discovery.Advertiser("Den", 47800, "172.20.1.73", p2p=True)
            ad._run()
            ad.set_p2p(True)            # (no change: no new announcement)
            ad.set_p2p(False)
        self.assertEqual([("-includeAWDL" in c) for c in cmds], [True, False])
