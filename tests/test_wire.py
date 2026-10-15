"""Protocol 2 on the wire: what 2.0 says to 1.x Macs and what it does with theirs.

Discovery stays cross-version (1.x and 2.0 still see each other), but they can't play together: a
1.x listener is kept connected with no audio, so its own window says "Different versions", and a 2.0
listener tells its person to update the 1.x host. Checked against the frozen 1.1.6 code itself."""
import importlib.util
import json
import os
import socket
import sys
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
sys.path.insert(0, here)
import syncaudio as sa  # noqa: E402
from test_syncaudio import free_port, wait_until  # noqa: E402

RATE = 48000


def legacy116():
    """SyncAudio 1.1.6 (tests/legacy/syncaudio116.py = git show e08f2b1:syncaudio.py)."""
    spec = importlib.util.spec_from_file_location("syncaudio116", os.path.join(here, "legacy", "syncaudio116.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def connect(port, hello, rcvbuf=None):
    sock = socket.socket()
    if rcvbuf:                                  # (loopback otherwise takes megabytes in)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    sock.settimeout(3)
    sock.connect(("127.0.0.1", port))
    sock.sendall(sa.frame(sa.HELLO, json.dumps({"magic": sa.MAGIC, **hello}).encode()))
    kind, payload = sa.recv_frame(sock)
    assert kind == sa.HEADER
    return sock, json.loads(payload)


class WireTest(unittest.TestCase):
    def setUp(self):
        self.port = free_port()
        self.logs = []
        self.version = mock.patch.object(sa, "APP_VERSION", "2.0.0")
        self.version.start()
        self.host = sa.Host(RATE, 2, 0.45, self.port, "Den", log=self.logs.append)
        self.host.start()

    def tearDown(self):
        self.host.stop()
        self.version.stop()

    def blocks(self, k):
        for i in range(k):
            self.host.on_block(np.full((480, 2), 0.1, np.float32), i * 480, time.monotonic())

    def test_discovery_answers_with_version_and_protocol(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(2)
            s.sendto(sa.DISCOVER, ("127.0.0.1", self.port))
            data, _ = s.recvfrom(512)
        self.assertEqual(data[:1], b"d")
        self.assertEqual(json.loads(data[1:]), {"name": "Den", "port": self.port, "version": "2.0.0", "proto": 2})
        self.assertEqual(sa.DISCOVER, b"Dsyncaudio/1")                 # what 1.x Macs send and answer

    def test_a_1x_listener_is_told_to_update_and_kept_connected_without_audio(self):
        sock, header = connect(self.port, {"name": "Sam", "version": "1.1.7", "codecs": ["opus"]})
        try:
            self.assertEqual((header["proto"], header["error"], header["version"]), (2, "update", "2.0.0"))
            self.assertEqual((header["rate"], header["channels"], header["codec"]), (RATE, 2, "pcm"))
            self.blocks(50)
            sock.settimeout(3)
            kind, payload = sa.recv_frame(sock)                         # no audio: an UPDATE every second
            self.assertEqual((kind, json.loads(payload)), (sa.UPDATE, {"error": "update"}))
            listeners = self.host.listeners()
            self.assertEqual([(l["name"], l["version"], l["old"]) for l in listeners], [("Sam", "1.1.7", True)])
            self.assertEqual([e["event"] for e in self.host.events], ["old_listener"])
            self.assertIn("! Sam has SyncAudio 1.1.7: it needs 2.0 to listen", self.logs)
        finally:
            sock.close()

    def test_the_real_1_1_6_listener_shows_different_versions(self):
        legacy = legacy116()
        opened = []
        output = types.SimpleNamespace(underflows=0, close=lambda: None)     # as much of 1.1.6's output as it uses
        client = legacy.Client("127.0.0.1", lambda rate, render: opened.append(rate) or output, self.port,
                               "Sam", log=lambda *a: None)
        client.start()
        try:
            # 1.1.6's window shows "Different versions" while connected and the versions differ
            self.assertTrue(wait_until(lambda: client.phase == "connected", 5), client.phase)
            self.assertEqual(client.host_version, "2.0.0")
            time.sleep(3.5)                                             # longer than its 3 s receive timeout
            self.assertEqual(client.phase, "connected")
            self.assertIsNone(client.player.read_n)                     # it never got anything to play
            self.assertEqual(opened, [RATE])
        finally:
            client.stop()

    def test_hello_says_what_this_mac_is(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen()
            client = sa.Client("127.0.0.1", mock.Mock(), server.getsockname()[1], "Sam", log=lambda *a: None,
                               hint=lambda: {"lead_ms_hint": 203.0, "transport": "blue"})
            client.start()
            try:
                server.settimeout(3)
                conn, _ = server.accept()
                with conn:
                    conn.settimeout(3)
                    kind, payload = sa.recv_frame(conn)
            finally:
                client.stop()
        hello = json.loads(payload)
        self.assertEqual(kind, sa.HELLO)
        self.assertEqual({k: hello[k] for k in ("magic", "name", "version", "proto", "engine", "lead_ms_hint",
                                                 "transport")},
                         {"magic": "syncaudio/1", "name": "Sam", "version": "2.0.0", "proto": 2,
                          "engine": "c-ring/2.0", "lead_ms_hint": 203.0, "transport": "blue"})

    def test_a_2_0_listener_tells_its_person_to_update_a_1x_host(self):
        legacy = legacy116()
        port = free_port()
        old = legacy.Host(RATE, 2, 0.25, port, "Kitchen", log=lambda *a: None)
        old.start()
        legacy.APP_VERSION = "1.1.6"
        open_output = mock.Mock()
        client = sa.Client("127.0.0.1", open_output, port, "Sam", log=lambda *a: None)
        client.start()
        try:
            self.assertTrue(wait_until(lambda: client.problem is not None, 5))
            self.assertEqual(client.problem["code"], "old_host")
            self.assertEqual(client.problem["message"],
                             "Kitchen has SyncAudio 1.1.6. Update it to 2.0 (Settings → Check for updates).")
            self.assertTrue(wait_until(lambda: client.phase == "retrying", 3))
            self.assertIn("Update it to 2.0", client.status())
            open_output.assert_not_called()                             # nothing opened, nothing played
            self.assertEqual([e["event"] for e in client.events], ["old_host"])
        finally:
            client.stop()
            old.stop()

    def test_probe_a_2_0_host(self):
        feeding = threading.Event()

        def feed():
            n = 0
            while not feeding.is_set():
                self.host.on_block(np.zeros((480, 2), np.float32), n, time.monotonic())
                n += 480
                time.sleep(0.01)
        threading.Thread(target=feed, daemon=True).start()
        try:
            result = sa.probe("127.0.0.1", self.port, seconds=1.0)
        finally:
            feeding.set()
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["host_name"], result["host_version"], result["proto"]), ("Den", "2.0.0", 2))


    def test_probe_a_1x_host_says_it_needs_updating(self):
        # it answers and streams to anyone: a reachable host, but "Click Join" would only say "update"
        legacy = legacy116()
        legacy.APP_VERSION = "1.1.6"
        port = free_port()
        old = legacy.Host(RATE, 2, 0.25, port, "Kitchen", log=lambda *a: None)
        old.start()
        feeding = threading.Event()

        def feed():
            while not feeding.is_set():
                old.on_block(np.full((480, 2), 0.1, np.float32), time.monotonic())
                time.sleep(0.01)
        threading.Thread(target=feed, daemon=True).start()
        try:
            result = sa.probe("127.0.0.1", port, seconds=1.0)
        finally:
            feeding.set()
            old.stop()
        self.assertFalse(result["ok"], result)
        self.assertGreater(result["audio_blocks_per_s"], 0)
        self.assertEqual((result["step"], result["problem"]["code"], result["host_version"]), ("version", "old_host", "1.1.6"))
        self.assertEqual(result["problem"]["message"],
                         "Kitchen is reachable, but has SyncAudio 1.1.6. Update it to 2.0 (Settings → Check for updates).")


class PeerTest(unittest.TestCase):
    def test_the_kernel_holds_little_and_says_how_much(self):
        port = free_port()
        host = sa.Host(RATE, 2, 0.45, port, "Den", log=lambda *a: None)
        host.start()
        sock, header = connect(port, {"name": "Sam", "proto": 2}, rcvbuf=4096)
        try:
            self.assertEqual(header.get("error"), None)
            self.assertTrue(wait_until(lambda: host.peers, 3))
            peer, = host.peers
            sndbuf = peer.sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            self.assertLessEqual(sndbuf, 2 * sa._Peer.SNDBUF)          # (Linux reports double what was set)
            for i in range(300):                                        # 3 s of audio the listener doesn't read
                host.on_block(np.full((480, 2), 0.1, np.float32), i * 480, time.monotonic() + 10)
            time.sleep(1.0)
            if sys.platform in ("darwin", "linux"):
                self.assertGreater(host.listeners()[0]["unsent"], 0)    # what the kernel holds, said
            threading.Thread(target=lambda: [sock.recv(65536) for _ in range(1000)], daemon=True).start()
            # the Wi-Fi "comes back": the send that waited is timed, so a stall shows up as one
            self.assertTrue(wait_until(lambda: host.listeners()[0]["send_stall_max_ms"] > 500, 5))
        finally:
            sock.close()
            host.stop()

    def test_slow_sends_are_timed(self):
        peer = sa._Peer(mock.Mock(sendall=lambda data: time.sleep(0.02)), "Sam", "10.0.0.2")
        peer.send(b"x")
        peer.send(b"y")
        self.assertGreaterEqual(peer.send_stall_max_ms(time.monotonic()), 20)
        self.assertEqual(peer.send_stall_max_ms(time.monotonic() + 10), 0)


if __name__ == "__main__":
    unittest.main()
