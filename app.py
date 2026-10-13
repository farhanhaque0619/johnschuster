#!/usr/bin/env python3
"""
SyncAudio, the app: a window around the syncaudio engine.

A tiny web server on 127.0.0.1 serves ui/index.html plus a JSON API, and a
native window (pywebview) shows it. Without pywebview it opens in the browser.

    python3 app.py              # the app
    python3 app.py --browser    # same, in the default browser
    python3 app.py --self-test  # check a packaged build and exit
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import http.server
import json
import os
import platform
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
from collections import deque
from pathlib import Path

import discovery
import room
import syncaudio as sa
import updater

APP_NAME = "SyncAudio"
VERSION = "1.1.3"
DOWNLOAD_URL = "https://github.com/farhanhaque0619/johnschuster/releases/latest"
BLACKHOLE_URL = "https://existential.audio/downloads/BlackHole2ch-0.7.1.pkg"
BLACKHOLE_SHA256 = "57b540f27a3e29c37e310e01bee0fdfab76733087e47f997ef9dccf851400dcf"
BLACKHOLE_PAGE = "https://existential.audio/blackhole/"
LINKS = {   # the only places the UI may ask us to open
    "microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
    "local_network": "x-apple.systempreferences:com.apple.preference.security?Privacy_LocalNetwork",
    "sound": "x-apple.systempreferences:com.apple.Sound-Settings.extension",
    "download": DOWNLOAD_URL,
    "blackhole": BLACKHOLE_PAGE,
}


FIREWALL = "/usr/libexec/ApplicationFirewall/socketfilterfw"


def parse_firewall(globalstate: str, blockall: str, app: str) -> dict:
    """Read socketfilterfw's answers: is the macOS firewall keeping other Macs out?"""
    g, b, a = globalstate.lower(), blockall.lower(), app.lower()
    enabled = ("enabled" in g or re.search(r"state = [12]", g) is not None) and "disabled" not in g
    block_all = bool(b.strip()) and "disabled" not in b and ("block all" in b or "enabled" in b)
    if "not part of the firewall" in a or not a.strip():
        app_state = "not_listed"
    elif "block" in a:
        app_state = "blocked"
    elif "permit" in a or "allow" in a:
        app_state = "allowed"
    else:
        app_state = "unknown"
    return {"enabled": enabled, "block_all": block_all, "app": app_state,
            "needs_fix": enabled and (block_all or app_state in ("blocked", "not_listed"))}


def firewall_status(app_path: str) -> dict:
    def ask(*args):
        try:
            done = subprocess.run([FIREWALL, *args], capture_output=True, text=True, timeout=5)
            return (done.stdout + done.stderr).strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    raw = {"global": ask("--getglobalstate"), "block_all": ask("--getblockall"),
           "app": ask("--getappblocked", app_path)}
    return {**parse_firewall(raw["global"], raw["block_all"], raw["app"]), "app_path": app_path, "raw": raw}


def resource_dir() -> Path:
    """Where ui/ and the bundled helper live (inside the .app when packaged)."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def bundled_switch_tool() -> str | None:
    tool = resource_dir() / "SwitchAudioSource"
    return str(tool) if tool.exists() else None


def default_prefs_file() -> Path:
    return Path.home() / "Library" / "Application Support" / APP_NAME / "prefs.json"


class NoNap:
    """Keeps macOS from napping the app while music plays. Behind Spotify the
    window is hidden, and after a minute or two App Nap slows the threads that
    move audio over Wi-Fi: it arrives later and later until it breaks up."""

    def __init__(self):
        self.token = None

    def hold(self, on: bool):
        if on == (self.token is not None):
            return
        try:
            import Foundation
            info = Foundation.NSProcessInfo.processInfo()
            if on:
                options = (getattr(Foundation, "NSActivityUserInitiated", 0x00FFFFFF | 1 << 20)
                           | getattr(Foundation, "NSActivityLatencyCritical", 0xFF00000000))
                self.token = info.beginActivityWithOptions_reason_(options, "Playing music in sync")
            else:
                info.endActivity_(self.token)
                self.token = None
        except Exception:           # not a Mac (tests), or no PyObjC: Info.plist still turns App Nap off
            self.token = None


class Controller:
    """Owns the current session (sharing or listening) and everything the UI shows."""

    def __init__(self, sd=None, prefs_file: Path | None = None, port: int = sa.PORT, scan: bool = True,
                 opener=None, switch_tool: str | None = None, room_port: int | None = None):
        self._sd = sd
        self.prefs_file = Path(prefs_file or default_prefs_file())
        self.prefs = self._load_prefs()
        self.settings = sa.Settings(float(self.prefs.get("offset_ms", 0.0)), float(self.prefs.get("volume", 1.0)),
                                    self.prefs.get("channel", "both"))
        self.name = self.prefs.get("name") or sa.computer_name()
        self.port = port
        self.opener = opener or (lambda url: subprocess.run(["open", url], check=False))
        self.switch_tool = switch_tool or bundled_switch_tool()
        self.session = None
        self.nonap = NoNap()
        self.error = None
        self.blackhole = None
        self.nearby: list = []
        self.idle_since = time.monotonic()
        self.install = {"state": "idle", "message": ""}
        self.build = updater.build_info(resource_dir(), VERSION)
        sa.APP_VERSION = self.build.get("version") or VERSION
        self.update_info = {"state": "idle", "available": False, "version": None, "message": ""}
        self.on_quit = None     # set by the window: closes the app
        self.events: deque = deque(maxlen=40)
        self.history: deque = deque(maxlen=180)     # one sample a second, for diagnostics
        self.scan_errors: dict = {}
        self.probe_result = None
        self.browser = None
        self._firewall, self._firewall_at = None, 0.0
        self.lock = threading.RLock()
        self._prefs_lock = threading.Lock()
        self.running = True
        self._recover_sound_output()
        self.device_id = self.prefs.get("device_id") or secrets.token_hex(8)
        if self.prefs.get("device_id") != self.device_id:
            self.prefs["device_id"] = self.device_id
            self._save_prefs()
        self.room = self.room_ad = self.room_browser = None
        self.room_note = None           # result of the last Connect, for the UI
        self.paused_at = None           # this Mac left the current round of music
        if room_port is None and scan:
            room_port = room.ROOM_PORT
        if room_port:
            self._start_room(room_port)
        if scan:
            threading.Thread(target=self._scan_loop, daemon=True).start()
            threading.Thread(target=self._update_loop, daemon=True).start()

    # ------------------------------------------------------------ plumbing
    @property
    def sd(self):
        if self._sd is None:
            try:
                import sounddevice
            except (ImportError, OSError) as e:
                raise sa.SetupError("audio", f"Couldn't load the audio library ({e}).")
            self._sd = sounddevice
        return self._sd

    def _load_prefs(self) -> dict:
        try:
            return json.loads(self.prefs_file.read_text())
        except (OSError, ValueError):
            return {}

    def _save_prefs(self):
        self.prefs.update(offset_ms=self.settings.offset_ms, volume=self.settings.volume,
                          channel=self.settings.channel)
        with self._prefs_lock:
            try:
                self.prefs_file.parent.mkdir(parents=True, exist_ok=True)
                self.prefs_file.write_text(json.dumps(self.prefs, indent=2))
            except (OSError, RuntimeError, TypeError) as e:
                self._log(f"couldn't save settings: {e}")

    def _log(self, msg: str):
        self.events.append({"t": time.time(), "msg": msg})
        print(msg, flush=True)

    def _recover_sound_output(self):
        """If the app died while sharing, the Mac's sound is still pointed at
        BlackHole. Put it back."""
        previous = self.prefs.pop("restore_output", None)
        if not previous:
            return
        system = sa.SystemOutput(self._log, self.switch_tool)
        if "blackhole" in system.current().lower():
            system.set(previous)
            self._log(f'Sound output put back to "{previous}".')
        self._save_prefs()

    def refresh_devices(self):
        """Re-read the audio device list (after installing BlackHole, say)."""
        sd = self.sd
        if self.session is None:
            sa.refresh_portaudio(sd)
        self.blackhole = any("blackhole" in d["name"].lower() for d in sd.query_devices())

    def _scan_loop(self):
        """While idle: ask the network who's sharing (broadcast, a chunk of the
        subnet, remembered addresses) and listen to Bonjour, AirDrop link included."""
        sweep, sweep_ip = None, None
        while self.running:
            self.nonap.hold(self.session is not None)
            if self.room_ad:        # the AirDrop link only while looking for each other, never while playing
                self.room_ad.set_p2p(self.session is None)
            if self.session is not None:
                for attr in ("browser", "room_browser"):    # the AirDrop link costs Wi-Fi latency
                    if getattr(self, attr):
                        getattr(self, attr).stop()
                        setattr(self, attr, None)
                time.sleep(1.0)
                continue
            if self.browser is None:
                self.browser = discovery.Browser().start()
            if self.room and self.room_browser is None:
                self.room_browser = discovery.Browser(service=discovery.ROOM_SERVICE).start()
            browser = self.browser
            ip = sa.lan_ip()
            if ip != sweep_ip:
                sweep_ip, sweep, broadcast = ip, None, None
                iface = discovery.interface_for(ip)
                if iface:
                    sweep, broadcast = discovery.Sweep(ip, iface[0]), iface[1]
            targets = {"255.255.255.255", *self.prefs.get("known_hosts", [])}
            if self.room_browser:
                targets.update(self.room_browser.found.values())
            if self.room:
                targets.update(m.get("ip") for m in self.room.members.values() if m.get("ip"))
            if sweep:
                targets.update(sweep.next_targets())
                targets.add(broadcast)
            targets.discard(None)
            # Macs with SyncAudio open (1.1+: can connect), asked in parallel with the sharing check.
            agents = []
            finder = threading.Thread(target=lambda: agents.extend(room.discover(targets, timeout=0.8)), daemon=True)
            if self.room:
                finder.start()
            by_name, errors = {}, []      # one entry per Mac, even when several routes found it
            replies = sa.discover_all(self.port, timeout=0.8, targets=targets, errors=errors)
            if self.room:
                finder.join(2)
            self.scan_errors = dict(collections.Counter(errors))
            for address, name in replies:
                if name not in by_name or by_name[name].startswith("127."):
                    by_name[name] = address
            for name, address in list(browser.found.items()):
                by_name.setdefault(name, address)
            nearby = {}
            for agent in agents:
                if agent["id"] == self.device_id:
                    continue
                self.room.saw(agent)
                nearby[agent["name"]] = {"name": agent["name"], "address": agent["address"], "id": agent["id"],
                                         "room_port": agent.get("room_port"), "sharing": bool(agent.get("sharing")),
                                         "member": agent["id"] in self.room.members}
            for name, address in by_name.items():      # sharing Macs, including older versions
                nearby.setdefault(name, {"name": name, "address": address, "sharing": True})["sharing"] = True
            if self.session is None:
                self.nearby = sorted((n for n in nearby.values() if not n.get("member")),
                                     key=lambda n: n["name"].lower())
            time.sleep(1.0)

    def _remember(self, addresses):
        """Macs we've played with: looked for directly next time."""
        known = self.prefs.get("known_hosts", [])
        fresh = [a for a in addresses if a and a not in known]
        if fresh:
            self.prefs["known_hosts"] = (fresh + known)[:10]
            self._save_prefs()

    # ------------------------------------------------------------- updates
    def _update_loop(self):
        time.sleep(5)
        while self.running:
            self.check_update()
            time.sleep(6 * 3600)

    def check_update(self):
        """Is the published build newer than this one?"""
        if self.update_info["state"] in ("downloading", "restarting"):
            return
        latest = updater.latest_release()
        if not latest:
            return
        mine = self.build.get("commit")
        available = bool(mine and latest["commit"] and latest["commit"] != mine)
        self.update_info = {**self.update_info, "available": available, "version": latest["version"],
                            "dmg_url": latest["dmg_url"], "checked": time.time()}

    def install_update(self):
        if self.update_info.get("dmg_url") and self.update_info["state"] not in ("downloading", "restarting"):
            self.update_info.update(state="downloading", message="Downloading the new version…")
            threading.Thread(target=self._install_update, daemon=True).start()

    def _install_update(self):
        try:
            bundle = updater.current_bundle()
            if not bundle:
                raise RuntimeError("this copy can't update itself; download the new version instead")
            staging = updater.staging_dir(bundle)
            new = updater.prepare(self.update_info["dmg_url"], staging)
            self.update_info.update(state="restarting", message="Restarting SyncAudio…")
            updater.apply(new, bundle, staging, os.getpid())
            time.sleep(0.5)
            if self.on_quit:
                self.on_quit()
        except Exception as e:
            self.update_info.update(state="failed", message=f"Couldn't update: {e}")

    # --------------------------------------------------------------- rooms
    def _start_room(self, port: int):
        agent = room.RoomAgent(
            me={"id": self.device_id, "name": self.name, "version": sa.APP_VERSION, "port": self.port},
            store=self.prefs.setdefault("room", {}), save=self._save_prefs, log=self._log,
            on_change=self._room_changed, port=port, my_ip=sa.lan_ip,
            sharing=lambda: isinstance(self.session, sa.HostSession))
        try:
            self.room = agent.start()
        except OSError as e:
            self._log(f"! connecting Macs is off: port {port} is in use ({e}). Is SyncAudio open twice?")
            return
        self.room_ad = discovery.Advertiser(self.name, port, sa.lan_ip(), service=discovery.ROOM_SERVICE,
                                            p2p_seconds=0).start()

    def _room_changed(self):
        threading.Thread(target=self._apply_room, daemon=True).start()

    def _apply_room(self):
        """Make this Mac do what the room says: share, listen to someone, or stop."""
        if not self.room or not self.running:
            return
        with self.lock:
            src = dict(self.room.source)
            sid, s = src["id"], self.session
            from_room = getattr(s, "from_room", False)
            if sid is None:
                if s and from_room:
                    self._stop_session()
                return
            if self.paused_at == src["at"]:         # this Mac left this round; wait for the next choice
                return
            if sid == self.device_id:
                if isinstance(s, sa.HostSession):
                    s.from_room = True
                    return
                if not self.share(from_room=True):
                    threading.Thread(target=self.room.set_source, args=(None,), daemon=True).start()
                return
            member = self.room.members.get(sid)
            if not member or not member.get("ip"):
                return
            port = member.get("port") or sa.PORT
            if isinstance(s, sa.JoinSession) and s.address == member["ip"] and s.port == port:
                s.from_room = True
                return
            self.join(member["ip"], port=port, from_room=True)

    def play(self, sid: str | None):
        """Whose music the room hears (None: nobody's)."""
        if self.room:
            self.paused_at = None
            self.error = None
            self.room.set_source(sid)

    def connect(self, address: str, room_port: int | None = None):
        if self.room and address:
            result = self.room.request_connect(address.strip(), int(room_port or room.ROOM_PORT))
            self.room_note = (f"Asked {result['name']} to connect. Click Accept on that Mac." if result["ok"]
                              else result["message"])

    def room_action(self, action: str, sid: str):
        if not self.room:
            return
        {"accept": self.room.accept, "decline": self.room.decline, "forget": self.room.forget}[action](sid)

    # ------------------------------------------------------------- actions
    def _stop_session(self):
        if self.session:
            self.session.stop()
            self.session = None
            if self.prefs.pop("restore_output", None):
                self._save_prefs()
        self.idle_since = time.monotonic()

    def _fail(self, e: Exception):
        code = getattr(e, "code", "error")
        self.error = {"code": code, "message": str(e)}
        if code == "no_blackhole":
            self.blackhole = False
        self._log(f"! {e}")

    def share(self, click: bool = False, from_room: bool = False) -> bool:
        if not from_room and not click and self.room and self.room.members:
            self.play(self.device_id)       # in a room, sharing means everyone switches to this Mac
            return True
        with self.lock:
            from_room = from_room or getattr(self.session, "from_room", False)
            self._stop_session()
            self.error = None
            try:
                self.refresh_devices()
                session = sa.HostSession(self.sd, self.name, delay_ms=self.prefs.get("delay_ms"), click=click,
                                         play_local=self.prefs.get("play_local", True), settings=self.settings,
                                         port=self.port, switch_tool=self.switch_tool, log=self._log)
                session.start()
            except Exception as e:
                self._fail(e)
                return False
            session.from_room = from_room
            self.session = session
            if session.system and session.system.previous:
                self.prefs["restore_output"] = session.system.previous
                self._save_prefs()
            return True

    def join(self, address: str, port: int | None = None, from_room: bool = False) -> bool:
        address = (address or "").strip()
        if not address:
            return False
        with self.lock:
            self._stop_session()
            self.error = None
            try:
                session = sa.JoinSession(self.sd, address, self.name, settings=self.settings,
                                         port=port or self.port, switch_tool=self.switch_tool, log=self._log)
                session.from_room = from_room
                self.session = session.start()
            except Exception as e:
                self._fail(e)
                return False
            if not from_room:
                self.prefs["last_host"] = address
                self._save_prefs()
            return True

    def stop(self):
        with self.lock:
            s = self.session
            if getattr(s, "from_room", False) and self.room:
                if isinstance(s, sa.HostSession):
                    self.play(None)                         # the music's source stops it for everyone
                    return
                self.paused_at = self.room.source["at"]     # a listener just steps out
            self._stop_session()
            self.error = None

    def update(self, data: dict):
        s = self.settings
        if "volume" in data:
            s.volume = min(1.0, max(0.0, float(data["volume"])))
        if "offset_ms" in data:
            s.offset_ms = min(500.0, max(-500.0, float(data["offset_ms"])))
        if data.get("channel") in ("both", "left", "right"):
            s.channel = data["channel"]
        if "group_volume" in data:
            if isinstance(self.session, sa.HostSession) and self.session.host:
                self.session.host.set_group_volume(data["group_volume"], by=self.name)
            elif isinstance(self.session, sa.JoinSession) and self.session.client:
                self.session.client.set_group_volume(data["group_volume"])
        if "name" in data and str(data["name"]).strip():
            self.name = self.prefs["name"] = str(data["name"]).strip()[:40]
        if "delay_ms" in data:
            delay = None if data["delay_ms"] is None else min(1500.0, max(40.0, float(data["delay_ms"])))
            self.prefs["delay_ms"] = delay
            if isinstance(self.session, sa.HostSession):
                self.session.set_delay(delay)
        restart = False
        if "play_local" in data:
            restart = bool(data["play_local"]) != self.prefs.get("play_local", True)
            self.prefs["play_local"] = bool(data["play_local"])
        self._save_prefs()
        if restart and isinstance(self.session, sa.HostSession):
            self.share(self.session.click)

    def install_blackhole(self):
        if self.install["state"] not in ("downloading", "installing"):
            self.install = {"state": "downloading", "message": "Downloading BlackHole…"}
            threading.Thread(target=self._install_blackhole, daemon=True).start()

    def _install_blackhole(self):
        try:
            pkg = Path(tempfile.mkdtemp()) / "BlackHole2ch.pkg"
            subprocess.run(["curl", "-fsSL", "--retry", "2", "-o", str(pkg), BLACKHOLE_URL],
                           check=True, capture_output=True, timeout=300)
            if hashlib.sha256(pkg.read_bytes()).hexdigest() != BLACKHOLE_SHA256:
                raise RuntimeError("the download didn't match the official BlackHole installer")
            self.install = {"state": "installing", "message": "Type your Mac password in the window that popped up."}
            path = str(pkg).replace("\\", "\\\\").replace('"', '\\"')
            script = (f'do shell script "/usr/sbin/installer -pkg " & quoted form of "{path}" & '
                      '" -target / && (/usr/bin/killall coreaudiod || true)" with administrator privileges '
                      'with prompt "SyncAudio wants to install BlackHole, the free audio driver that lets it '
                      'hear what this Mac plays."')
            done = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            if done.returncode != 0:
                if "-128" in done.stderr or "canceled" in done.stderr.lower():
                    self.install = {"state": "idle", "message": "Install cancelled."}
                    return
                raise RuntimeError(done.stderr.strip() or "the installer failed")
            for _ in range(10):         # CoreAudio takes a moment to come back
                time.sleep(1)
                self.refresh_devices()
                if self.blackhole:
                    break
            if self.blackhole:
                self.install = {"state": "done", "message": "BlackHole is installed."}
                if self.error and self.error["code"] == "no_blackhole":
                    self.error = None
            else:
                self.install = {"state": "restart", "message": "Installed. Restart your Mac to finish setting it up."}
        except Exception as e:
            self.install = {"state": "failed", "message": f"Couldn't install BlackHole: {e}"}

    def open(self, what: str):
        if what in LINKS:
            self.opener(LINKS[what])

    def copy(self, text: str, limit: int | None = 500):
        """Put text on the clipboard. The page may only copy short things (an address)."""
        try:
            subprocess.run(["pbcopy"], input=str(text)[:limit], text=True, check=False, timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass

    def firewall(self, fresh: bool = False):
        """The macOS firewall's view of this app (cached; it's three subprocesses)."""
        if sys.platform != "darwin" or not os.path.exists(FIREWALL):
            return None
        if fresh or self._firewall is None or time.monotonic() - self._firewall_at > 20:
            self._firewall = firewall_status(str(updater.current_bundle() or sys.executable))
            self._firewall_at = time.monotonic()
        return self._firewall

    def fix_firewall(self):
        """Let other Macs connect: add SyncAudio to the firewall as allowed (asks for the password)."""
        fw = self.firewall(fresh=True)
        if not fw or not fw["needs_fix"]:
            return
        path = fw["app_path"].replace("\\", "\\\\").replace('"', '\\"')
        extra = f' & "; {FIREWALL} --setblockall off"' if fw["block_all"] else ""
        done = subprocess.run(
            ["osascript", "-e", f'set p to "{path}"', "-e",
             f'do shell script "{FIREWALL} --add " & quoted form of p & "; {FIREWALL} --unblockapp " & '
             f'quoted form of p{extra} with administrator privileges with prompt '
             '"SyncAudio wants to let your other Macs connect to it through the macOS firewall."'],
            capture_output=True, text=True)
        if done.returncode != 0 and "-128" not in done.stderr:
            self._log(f"! couldn't change the firewall: {done.stderr.strip()}")
        self.firewall(fresh=True)

    def test_connection(self, address: str):
        address = (address or "").strip()
        if not address:
            return
        self.probe_result = {"address": address, "running": True}
        def run():
            self.probe_result = sa.probe(address, self.port)
        threading.Thread(target=run, daemon=True).start()

    def diagnostics(self) -> str:
        """Everything useful for working out what's wrong, as text to paste."""
        try:
            firewall = subprocess.run(["/usr/libexec/ApplicationFirewall/socketfilterfw", "--getglobalstate"],
                                      capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            firewall = None
        try:
            devices = [{"name": d["name"], "in": d["max_input_channels"], "out": d["max_output_channels"]}
                       for d in self.sd.query_devices()]
        except Exception as e:
            devices = str(e)
        ip = sa.lan_ip()
        iface = discovery.interface_for(ip)
        report = {
            "app": self.build, "macos": platform.mac_ver()[0], "machine": platform.machine(),
            "network": {"ip": ip, "subnet": str(iface[0]) if iface else None, "firewall": firewall,
                        "firewall_app": self.firewall(),
                        "scan_send_errors": self.scan_errors, "known_hosts": self.prefs.get("known_hosts", []),
                        "nearby": self.nearby, "bonjour": bool(discovery.dns_sd())},
            "devices": devices,
            "session": self.session.info() if self.session else None,
            "room": self.room.status() if self.room else None,
            "settings": self.state()["settings"],
            "history": list(self.history),
            "events": list(self.events),
            "probe": self.probe_result,
        }
        return json.dumps(report, indent=1, default=str)

    def summary(self, report: dict) -> str:
        """The few numbers that matter, short enough to paste into a chat."""
        s = report.get("session") or {}
        lines = [f"SyncAudio {report['app'].get('version')} ({(report['app'].get('commit') or 'dev')[:7]}) "
                 f"macOS {report['macos']}",
                 f"network {report['network']['ip']} {report['network']['subnet']} "
                 f"firewall {'on' if (report['network'].get('firewall_app') or {}).get('enabled') else 'off'}",
                 f"mode {s.get('mode', 'idle')} delay {s.get('delay_ms')} ms output {s.get('output')}"]
        if s.get("mode") == "hosting":
            lines.append(f"capture glitches {s.get('capture_glitches')} level {s.get('level')} silent {s.get('silent_s')} s")
            lines.append(f"this Mac {s.get('local')}")
            lines += [f"listener {l.get('name')}: {l.get('state')} err {l.get('err_ms')} wifi {l.get('wifi_ms')} "
                      f"headroom {l.get('headroom_ms')} dropouts {l.get('dropouts')} resyncs {l.get('resyncs')} "
                      f"underflows {l.get('underflows')} late {l.get('late')} speed {l.get('speed_pct')}% "
                      f"v{l.get('version')}" for l in s.get("listeners", [])]
        elif s:
            lines.append(f"listening to {s.get('host_name')} v{s.get('host_version')}: {s.get('state')} err {s.get('err_ms')} "
                         f"wifi {s.get('wifi_ms')} headroom {s.get('headroom_ms')} dropouts {s.get('dropouts')} "
                         f"resyncs {s.get('resyncs')} underflows {s.get('underflows')} speed {s.get('speed_pct')}% "
                         f"silent {s.get('incoming_silent_s')}")
        history = report.get("history", [])[-30:]
        if history:
            keys = ("state", "err_ms", "headroom_ms", "dropouts", "delay_ms")
            def compact(h):
                src = h.get("local") or h
                return "/".join(str(src.get(k, h.get(k))) for k in keys)
            lines.append("last 30 s (state/err/headroom/dropouts/delay): " + " ".join(compact(h) for h in history[::3]))
        lines += [f"event {e['msg']}" for e in report.get("events", [])[-8:]]
        return "\n".join(lines)

    def copy_diagnostics(self):
        full = self.diagnostics()
        report = json.loads(full)
        path = Path.home() / "Desktop" / "SyncAudio diagnostics.txt"
        try:
            path.write_text(full)
            where = f"\nfull report: {path}"
        except OSError:
            where = ""
        self.copy(self.summary(report) + where, limit=None)

    def shutdown(self):
        self.running = False
        for thing in (self.room_browser, self.room_ad, self.room):
            if thing:
                thing.stop()
        if self.browser:            # its dns-sd would outlive the app otherwise
            self.browser.stop()
            self.browser = None
        with self.lock:
            self._stop_session()

    # --------------------------------------------------------------- state
    def state(self) -> dict:
        if self.blackhole is None:
            try:
                self.refresh_devices()
            except Exception as e:
                self._fail(e)
        session = self.session
        info = session.info() if session else None
        if info and (not self.history or time.time() - self.history[-1]["t"] >= 1):
            sample = {"t": round(time.time(), 1), "mode": info["mode"], "delay_ms": info.get("delay_ms")}
            if info["mode"] == "hosting":
                sample["local"] = info["local"]
                sample["listeners"] = [{k: l.get(k) for k in ("name", "state", "err_ms", "headroom_ms",
                                                              "wifi_ms", "dropouts", "speed_pct")}
                                       for l in info["listeners"]]
            else:
                sample.update({k: info.get(k) for k in ("connection", "state", "err_ms", "headroom_ms",
                                                        "wifi_ms", "dropouts", "output", "speed_pct")})
            self.history.append(sample)
        if info and info["mode"] == "listening" and info["connection"] == "connected":
            self._remember([info["address"]])
        elif info and info["mode"] == "hosting":
            self._remember([l["ip"] for l in info["listeners"]])
        previous = getattr(getattr(session, "system", None), "previous", None)
        if isinstance(session, sa.HostSession) and previous and self.prefs.get("restore_output") != previous:
            self.prefs["restore_output"] = previous
            self._save_prefs()
        s = self.settings
        return {
            "app": {"name": APP_NAME, "version": self.build.get("version") or VERSION, "download_url": DOWNLOAD_URL},
            "update": {k: v for k, v in self.update_info.items() if k != "dmg_url"},
            "name": self.name,
            "address": sa.lan_ip(),
            "mode": info["mode"] if info else "idle",
            "session": info,
            "error": self.error,
            "blackhole": self.blackhole,
            "install": self.install,
            "nearby": self.nearby,
            "idle_s": round(time.monotonic() - self.idle_since, 1),
            "last_host": self.prefs.get("last_host"),
            "settings": {"volume": s.volume, "offset_ms": s.offset_ms, "channel": s.channel,
                         "delay_ms": self.prefs.get("delay_ms"), "play_local": self.prefs.get("play_local", True)},
            "events": list(self.events)[-6:],
            "probe": self.probe_result,
            "device_id": self.device_id,
            "room": {**self.room.status(), "note": self.room_note,
                     "paused": self.paused_at is not None and self.paused_at == self.room.source["at"]}
                    if self.room else None,
            "firewall": self.firewall() if info and info["mode"] == "hosting" else None,
            "scan_blocked": self.scan_errors.get(65, 0) > 0,     # EHOSTUNREACH: Local Network permission
        }


# ------------------------------------------------------------------ web API
def make_server(controller: Controller, port: int = 0) -> http.server.ThreadingHTTPServer:
    token = secrets.token_urlsafe(24)
    page = (resource_dir() / "ui" / "index.html").read_text(encoding="utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _allowed(self, need_token: bool) -> bool:
            # Only this machine's UI: right Host header (stops DNS rebinding) and the secret token.
            host_ok = self.headers.get("Host") in (f"127.0.0.1:{server.server_port}", f"localhost:{server.server_port}")
            if not host_ok or (need_token and not secrets.compare_digest(self.headers.get("X-Token", ""), token)):
                self.send_error(403)
                return False
            return True

        def _send(self, body: bytes, kind: str):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _state(self):
            self._send(json.dumps(controller.state()).encode(), "application/json")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                if self._allowed(need_token=False):
                    self._send(page.replace("__TOKEN__", token).encode(), "text/html; charset=utf-8")
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif path == "/api/state":
                if self._allowed(need_token=True):
                    self._state()
            else:
                self.send_error(404)

        def do_POST(self):
            if not self._allowed(need_token=True):
                return
            try:
                length = min(int(self.headers.get("Content-Length") or 0), 65536)
                data = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self.send_error(400)
            actions = {
                "/api/share": lambda: controller.share(bool(data.get("click"))),
                "/api/join": lambda: controller.join(str(data.get("address", ""))),
                "/api/play": lambda: controller.play(data.get("id")),
                "/api/connect": lambda: controller.connect(str(data.get("address", "")), data.get("room_port")),
                "/api/room": lambda: controller.room_action(str(data.get("action")), str(data.get("id"))),
                "/api/stop": controller.stop,
                "/api/settings": lambda: controller.update(data),
                "/api/install-blackhole": controller.install_blackhole,
                "/api/check-devices": controller.refresh_devices,
                "/api/open": lambda: controller.open(str(data.get("what"))),
                "/api/copy": lambda: controller.copy(str(data.get("text", ""))),
                "/api/dismiss": lambda: setattr(controller, "error", None),
                "/api/check-update": controller.check_update,
                "/api/test-connection": lambda: controller.test_connection(str(data.get("address", ""))),
                "/api/copy-diagnostics": controller.copy_diagnostics,
                "/api/fix-firewall": controller.fix_firewall,
                "/api/install-update": controller.install_update,
            }
            action = actions.get(self.path)
            if not action:
                return self.send_error(404)
            try:
                action()
            except Exception as e:
                controller._fail(e)
            self._state()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def self_test() -> int:
    """Used by the build: proves the packaged app has everything it needs."""
    import numpy
    import sounddevice
    print("numpy", numpy.__version__, "| sounddevice", sounddevice.__version__,
          "|", sounddevice.get_portaudio_version()[1])
    print("audio devices:", len(sounddevice.query_devices()))
    try:
        import webview
        print("pywebview", getattr(webview, "__version__", "ok"))
    except ImportError as e:
        print("pywebview missing:", e)
        return 1
    print("SwitchAudioSource:", bundled_switch_tool() or "not bundled")
    with tempfile.TemporaryDirectory() as tmp:
        controller = Controller(sd=sounddevice, prefs_file=Path(tmp) / "prefs.json", scan=False)
        server = make_server(controller)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_port
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        token = page.split('data-token="')[1].split('"')[0]
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/state", headers={"X-Token": token})
        state = json.loads(urllib.request.urlopen(req).read())
        server.shutdown()
        server.server_close()
    print("UI served, state mode:", state["mode"], "| BlackHole installed:", state["blackhole"])
    print("self-test ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog=APP_NAME)
    ap.add_argument("--browser", action="store_true", help="open in the browser instead of a window")
    ap.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--check-update", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--apply-update", metavar="DMG", help=argparse.SUPPRESS)
    ap.add_argument("--no-relaunch", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--probe", metavar="ADDRESS", help="test the connection to a sharing Mac")
    ap.add_argument("--port", type=int, default=sa.PORT, help=argparse.SUPPRESS)
    ap.add_argument("--firewall-status", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cli", nargs=argparse.REMAINDER, help="run the Terminal version (e.g. --cli host --click)")
    args, _ = ap.parse_known_args(argv)     # Finder may add its own arguments
    if args.cli is not None:
        return sa.main(args.cli)
    if args.firewall_status:    # used by the build to check how macOS answers
        print(json.dumps(firewall_status(str(updater.current_bundle() or sys.executable))))
        return
    if args.probe:
        result = sa.probe(args.probe, args.port)
        print(json.dumps(result, indent=1))
        sys.exit(0 if result["ok"] else 1)
    if args.self_test:
        sys.exit(self_test())
    if args.check_update:   # used by the build to test the real thing
        print(json.dumps({"this": updater.build_info(resource_dir(), VERSION), "latest": updater.latest_release()}))
        return
    if args.apply_update:
        bundle = updater.current_bundle()
        staging = updater.staging_dir(bundle)
        new = updater.prepare(args.apply_update, staging)
        updater.apply(new, bundle, staging, os.getpid(), relaunch=not args.no_relaunch)
        print("update staged; swapping after exit")
        return

    sys.setswitchinterval(0.001)    # hand Python to the audio threads sooner: fewer clicks
    controller = Controller()
    server = make_server(controller)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"
    webview = None
    if not args.browser:
        try:
            import webview
        except ImportError:
            pass
    result = {"ok": not args.smoke_test}
    try:
        if webview:
            window = webview.create_window(APP_NAME, url, width=460, height=780, min_size=(400, 600))
            controller.on_quit = window.destroy
            if args.smoke_test:     # the build opens the real window once, checks the UI drew, and quits
                def check():
                    time.sleep(4)
                    shown = window.evaluate_js("!document.getElementById('view-idle').hidden && "
                                               "document.getElementById('this-mac').textContent")
                    print("window shows:", shown, flush=True)
                    result["ok"] = bool(shown)
                    window.destroy()
                window.events.loaded += lambda: threading.Thread(target=check, daemon=True).start()
                guard = threading.Timer(30, window.destroy)     # never hang the build
                guard.daemon = True
                guard.start()
            webview.start()
        else:
            webbrowser.open(url)
            print(f"{APP_NAME} is running at {url}  (Ctrl+C to quit)")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()
        server.shutdown()
        server.server_close()
    if not result["ok"]:
        sys.exit("smoke test: the window didn't show the app")


if __name__ == "__main__":
    main()
