"""Rooms: Macs that stay connected and switch whose music plays."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import app  # noqa: E402
import room  # noqa: E402
import syncaudio as sa  # noqa: E402
from test_syncaudio import FakeSoundDevice, free_port, wait_until  # noqa: E402


def agent(name, changes=None):
    store = {}
    a = room.RoomAgent({"id": name.lower(), "name": name, "version": "test", "port": free_port()}, store,
                       log=lambda *x: None, on_change=(lambda: changes.append(name)) if changes is not None else None,
                       port=free_port(), my_ip=lambda: "127.0.0.1")
    return a.start()


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.agents = []

    def tearDown(self):
        for a in self.agents:
            a.stop()

    def pair(self, *names):
        agents = [agent(n) for n in names]
        self.agents += agents
        return agents

    def connect(self, asker, owner):
        self.assertTrue(asker.request_connect("127.0.0.1", owner.port)["ok"])
        self.assertTrue(wait_until(lambda: asker.me["id"] in owner.requests, 5))
        self.assertTrue(owner.accept(asker.me["id"]))

    def test_connect_needs_the_other_side_to_accept(self):
        a, b = self.pair("Farhan", "Saifan")
        a.request_connect("127.0.0.1", b.port)
        self.assertTrue(wait_until(lambda: "farhan" in b.requests, 5))
        self.assertEqual(a.members, {}, "nothing happens until Saifan accepts")
        self.assertEqual(b.status()["requests"][0]["name"], "Farhan")
        b.accept("farhan")
        self.assertEqual(set(a.members), {"saifan"})
        self.assertEqual(set(b.members), {"farhan"})
        self.assertEqual(a.store["key"], b.store["key"])
        self.assertTrue(wait_until(lambda: a.online("saifan") and b.online("farhan"), 6))

    def test_either_mac_switches_the_music_for_both(self):
        a, b = self.pair("Farhan", "Saifan")
        self.connect(a, b)
        a.set_source("farhan")
        self.assertTrue(wait_until(lambda: b.source["id"] == "farhan", 5))
        b.set_source("saifan")
        self.assertTrue(wait_until(lambda: a.source["id"] == "saifan", 5))
        self.assertEqual(a.status()["source"], {"id": "saifan", "name": "Saifan", "by": "Saifan"})
        a.set_source(None)
        self.assertTrue(wait_until(lambda: b.source["id"] is None, 5))

    def test_strangers_cannot_switch_your_music(self):
        a, b = self.pair("Farhan", "Saifan")
        self.connect(a, b)
        stranger = {"type": "sync", "magic": room.MAGIC, "from": {"id": "farhan", "name": "Farhan"},
                    "ts": time.time(), "source": {"id": "saifan", "at": time.time() + 999, "by": "x"},
                    "members": {}, "mac": "0" * 64}
        reply = room.send("127.0.0.1", b.port, stranger)
        self.assertFalse(reply["ok"])
        self.assertIsNone(b.source["id"], "a forged message can't make Saifan's Mac start sharing")
        welcome = {"type": "welcome", "magic": room.MAGIC, "from": {"id": "evil", "name": "Evil"},
                   "ts": time.time(), "key": "00" * 32, "members": {}, "source": {}}
        self.assertFalse(room.send("127.0.0.1", a.port, welcome)["ok"], "unasked welcomes are ignored")
        self.assertEqual(set(a.members), {"saifan"})

    def test_three_macs_end_up_in_one_room(self):
        a, b, c = self.pair("Farhan", "Saifan", "Alex")
        self.connect(a, b)
        self.connect(c, b)              # Alex joins through Saifan; Farhan learns about Alex
        self.assertTrue(wait_until(lambda: "alex" in a.members and a.online("alex"), 10), a.members)
        c.set_source("farhan")
        self.assertTrue(wait_until(lambda: a.source["id"] == b.source["id"] == "farhan", 6))

    def test_forget_disconnects_both_sides(self):
        a, b = self.pair("Farhan", "Saifan")
        self.connect(a, b)
        a.forget("saifan")
        self.assertEqual(a.members, {})
        self.assertTrue(wait_until(lambda: b.members == {}, 5))

    def test_discovery_finds_open_apps(self):
        a, = self.pair("Farhan")
        found = room.discover(["127.0.0.1"], timeout=0.5, ports=[a.port])
        self.assertEqual([(f["id"], f["name"], f["sharing"]) for f in found], [("farhan", "Farhan", False)])


class TwoMacsTest(unittest.TestCase):
    """Two whole apps (simulated audio) connect once, then pass the music back and forth."""

    def setUp(self):
        self.which = mock.patch.object(sa.shutil, "which", return_value=None)
        self.which.start()
        tmp = Path(tempfile.mkdtemp())
        self.macs = []
        for name in ("Farhan", "Saifan"):
            c = app.Controller(sd=FakeSoundDevice(), prefs_file=tmp / f"{name}.json", port=free_port(),
                               scan=False, room_port=free_port())
            c.name = name
            self.macs.append(c)

    def tearDown(self):
        for c in self.macs:
            c.shutdown()
        self.which.stop()

    def test_switch_whose_music_without_stopping(self):
        farhan, saifan = self.macs
        farhan.connect("127.0.0.1", saifan.room.port)
        self.assertIn("Click Accept", farhan.room_note)
        self.assertTrue(wait_until(lambda: saifan.state()["room"]["requests"], 5))
        saifan.room_action("accept", farhan.device_id)
        self.assertTrue(wait_until(lambda: farhan.state()["room"]["members"]
                                   and farhan.state()["room"]["members"][0]["online"], 6))

        def modes():
            return farhan.state()["mode"], saifan.state()["mode"]

        def synced(c):
            info = c.state()["session"]
            return info and (info.get("state") == "in_sync" or (info.get("local") or {}).get("state") == "in_sync")

        farhan.play(farhan.device_id)           # Farhan's music
        self.assertTrue(wait_until(lambda: modes() == ("hosting", "listening") and synced(saifan), 15), modes())
        self.assertEqual(saifan.state()["session"]["host_name"], "Farhan")

        saifan.play(saifan.device_id)           # Saifan switches to his own music, nobody stops anything
        self.assertTrue(wait_until(lambda: modes() == ("listening", "hosting") and synced(farhan), 15), modes())
        self.assertEqual(farhan.state()["session"]["host_name"], "Saifan")

        farhan.play(farhan.device_id)           # and back
        self.assertTrue(wait_until(lambda: modes() == ("hosting", "listening") and synced(saifan), 15), modes())

        farhan.stop()                           # the Mac playing stops the music for everyone
        self.assertTrue(wait_until(lambda: modes() == ("idle", "idle"), 8), modes())

    def test_a_listener_can_step_out(self):
        farhan, saifan = self.macs
        farhan.connect("127.0.0.1", saifan.room.port)
        wait_until(lambda: saifan.state()["room"]["requests"], 5)
        saifan.room_action("accept", farhan.device_id)
        farhan.play(farhan.device_id)
        self.assertTrue(wait_until(lambda: saifan.state()["mode"] == "listening", 10))
        saifan.stop()                           # Saifan leaves; Farhan keeps playing
        time.sleep(3)
        self.assertEqual((farhan.state()["mode"], saifan.state()["mode"]), ("hosting", "idle"))
        self.assertTrue(saifan.state()["room"]["paused"])
        farhan.play(saifan.device_id)           # a new choice brings Saifan back in
        self.assertTrue(wait_until(lambda: saifan.state()["mode"] == "hosting", 10))


if __name__ == "__main__":
    unittest.main()
