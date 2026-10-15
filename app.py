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
import gc
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

import audio_engine as ae
import discovery
import room
import sacore
import syncaudio as sa
import updater

APP_NAME = "SyncAudio"
VERSION = "2.0.0"
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
        try:     # close_fds=False and a full path: started with posix_spawn, not a fork of the whole app
            done = subprocess.run([FIREWALL, *args], capture_output=True, text=True, timeout=5, close_fds=False)
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


def coreaudio_overloads(text: str | None = None, limit: int = 50) -> list | None:
    """macOS's own log of moments the sound system skipped audio ("overload"): the static,
    as macOS saw it. Only read when diagnostics are copied (it takes a few seconds)."""
    if text is None:
        try:
            text = subprocess.run(["log", "show", "--last", "15m", "--info", "--style", "compact", "--predicate",
                                   'process == "coreaudiod" AND (eventMessage CONTAINS[c] "overload" '
                                   'OR eventMessage CONTAINS[c] "skipping cycle")'],
                                  capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    lines = [l.strip() for l in text.splitlines() if "overload" in l.lower() or "skipping cycle" in l.lower()]
    return [f"{len(lines)} in the last 15 min"] + lines[-limit:]


def system_info(run=None) -> dict:
    """The Mac itself: its model, Low Power Mode, and whether it is slowing down to cool off (both make
    every thread late, the audio threads too). Only read when diagnostics are copied."""
    def ask(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=5, close_fds=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    run = run or ask
    power = run("/usr/bin/pmset", "-g") or ""
    low = [line.split()[-1] for line in power.splitlines() if line.split()[:1] == ["lowpowermode"]]
    return {"model": run("/usr/sbin/sysctl", "-n", "hw.model"), "low_power_mode": low[0] if low else None,
            "thermal": run("/usr/bin/pmset", "-g", "therm")}


def wifi_info(text: str | None = None) -> dict | None:
    """The Wi-Fi network this Mac is on: channel, band, signal. On 2.4 GHz, Wi-Fi and
    Bluetooth headphones fight over the same radio, which sounds like static."""
    if text is None:
        try:
            text = subprocess.run(["system_profiler", "SPAirPortDataType", "-json"], capture_output=True,
                                  text=True, timeout=15).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    try:
        data = json.loads(text)
    except ValueError:
        return None

    def find(node):
        if isinstance(node, dict):
            if isinstance(node.get("spairport_current_network_information"), dict):
                return node["spairport_current_network_information"]
            node = list(node.values())
        if isinstance(node, list):
            for item in node:
                found = find(item)
                if found:
                    return found
        return None

    net = find(data)
    if not net:
        return None
    channel = str(net.get("spairport_network_channel", ""))
    m = re.search(r"\b(2|5|6)\s*GHz", channel)
    number = re.match(r"\s*(\d+)", channel)
    band = f"{'2.4' if m.group(1) == '2' else m.group(1)} GHz" if m else (
        "2.4 GHz" if number and int(number.group(1)) <= 14 else None)
    return {"channel": channel or None, "band": band, "phy": net.get("spairport_network_phymode"),
            "signal": net.get("spairport_signal_noise")}


# What a second of history keeps (spec 11.2). A listener's row on the sharing Mac is short: that Mac's
# own diagnostics have the rest - but which output it played on, second by second, is here too (1.x's
# never said). A listening Mac keeps all of its numbers but these.
PEER_KEYS = ("id", "name", "out", "uid", "state", "err_ms", "err_max_ms", "headroom_ms", "spare_1s_ms", "spare_30_ms",
             "wifi_ms", "dropouts", "underruns", "holes", "skips", "skipped_ms", "overloads", "io", "lead_ms",
             "dll_ppm", "lossy", "producer_max_ms", "gc_max_ms", "gil_probe_max_ms", "queue", "unsent", "late",
             "send_stall_max_ms")
NOT_SAMPLED = {"mode", "address", "host_name", "host_version", "error", "problem", "group_volume",
               "group_volume_by", "audio", "clock_pings", "events", "clock", "synced", "engine"}
DELAY_EVENTS = ("delay_change", "glide_done", "lower_abandoned")


def io_text(io) -> str:
    """Frames per IO cycle: 2048, or 1024-2048 when it varied."""
    lo, hi = io or (None, None)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def device_facts(d: dict) -> dict:
    """An audio device for diagnostics: PortAudio's description, and with the macOS helper, CoreAudio's
    own figures (in frames): its UID, how it's connected, its latencies and its buffer range."""
    out = {"name": d["name"], "in": d["max_input_channels"], "out": d["max_output_channels"],
           "rate": d.get("default_samplerate"),
           # PortAudio's latency range: the low end is the device's own latency plus its smallest buffer
           "out_latency_ms": [round(d.get("default_low_output_latency", 0) * 1000, 1),
                              round(d.get("default_high_output_latency", 0) * 1000, 1)]
           if d["max_output_channels"] else None}
    facts = {kind: sacore.mac_call("device_info", d["name"], kind == "out") for kind in ("out", "in") if out[kind]}
    facts = {kind: f for kind, f in facts.items() if f}
    out["uid"] = next((f["uid"] for f in facts.values()), None)
    out["coreaudio"] = {kind: {k: f.get(k) for k in ("transport", "latency", "safety", "stream_latency",
                                                     "buffer_frames", "buffer_min", "buffer_max", "nominal_rate")}
                        for kind, f in facts.items()} or None
    return out


def grew(values) -> int:
    """How much a running count went up. One that starts again from 0 (a reopened output) didn't drop."""
    values = [v for v in values if v is not None]
    return sum(max(0, b - a) for a, b in zip(values, values[1:]))


def player_summary(label: str, p: dict, past: list, delay: str) -> list:
    """A player's lines in the summary (spec 11.4). past: (t, its numbers) from the history, oldest first.
    "static check" is the one to read first: skipped cycles and overloads are macOS dropping audio on
    that output. 0 / 0 while static is still heard points at the Bluetooth link itself; counts while
    SyncAudio's callback stays tiny (cb_max_us) point at macOS, and the output climbs to bigger buffers."""
    errs = [e for e in [h.get("err_max_ms") for _, h in past if h] + [p.get("err_max_ms")] if e is not None]
    lossy = f" LOSSY {p.get('loss_pct')} %" if p.get("lossy") else ""
    lines = [f"{label}: out {p.get('out')} io {io_text(p.get('io'))} "
             f"lead {p.get('lead_ms')} ms | skips {p.get('skips')} ({p.get('skipped_ms')} ms) "
             f"overloads {p.get('overloads')} underruns {p.get('underruns')} | "
             f"err max {max(errs, default=None)} ms | spare30 {p.get('spare_30_ms')} ms | "
             f"dll {p.get('dll_ppm')} ppm jit {p.get('dll_jitter_ms')} ms{lossy} | {delay}"]
    recent = [(t, h) for t, h in past if h and t >= past[-1][0] - 60] if past else []
    if recent:
        counts = {k: grew([h.get(k) for _, h in recent] + [p.get(k)]) for k in ("skips", "overloads")}
        lines.append(f"  static check: {counts['skips']} skipped cycles / {counts['overloads']} overloads "
                     f"in the last {max(1, round(past[-1][0] - recent[0][0]))} s")
    return lines


def selftest_summary(r: dict) -> str:
    v2, old = r["v2"], r.get("legacy")
    then = f" | old way io {old['frames_min']}-{old['frames_max']} lost {old['lost_ms_per_s']} ms/s" if old else ""
    return (f"headphone test on {r['output']}: {r['verdict']}{then} | 2.0 io {v2['io_frames']} skips {v2['skips']} "
            f"({v2['skipped_ms']} ms) overloads {v2['overloads']} underruns {v2['underruns']} "
            f"cb max {v2['cb_max_us']} us{' LOSSY' if v2.get('lossy') else ''}")


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
        if "devices" not in self.prefs and "offset_ms" in self.prefs:
            # 1.x's one sync nudge made up for 1.x's own timing, which drifted late; 2.0 plays every sample
            # on time, so each output starts again from 0 (the old value is kept for diagnostics)
            self.prefs["offset_ms_1x"] = self.prefs.pop("offset_ms")
        # Every output this Mac played on, by CoreAudio UID (its name without the macOS helper): name, uid,
        # offset_ms (its own sync nudge), io_level (bigger buffers it needed), lead_ms (IOProc to ear), seen.
        devices = self.prefs.setdefault("devices", {})
        self.prefs.setdefault("peer_leads", {})     # the same lead for the Macs that listened to this one
        self.settings = sa.Settings(float(self.prefs.get("offset_ms", 0.0)), float(self.prefs.get("volume", 1.0)),
                                    self.prefs.get("channel", "both"),
                                    {k: d["offset_ms"] for k, d in devices.items() if "offset_ms" in d})
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
        self.probes = ()                # GC and GIL timing, while a session runs
        self.selftest = {"running": False}      # "Test my headphones": progress, then the result
        self._selftest = None
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
        s = self.settings
        devices = self.prefs.setdefault("devices", {})
        for key, ms in list(s.offsets.items()):    # each output's nudge, kept with what else is known about it
            devices.setdefault(key, {})["offset_ms"] = ms
        self.prefs.update(offset_ms=s.default_offset, volume=s.volume, channel=s.channel)
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
        subnet, remembered addresses) and listen to Bonjour."""
        sweep, sweep_ip = None, None
        while self.running:
            self.nonap.hold(self.session is not None)
            if self.session is not None:
                for attr in ("browser", "room_browser"):    # browsing costs a little Wi-Fi traffic
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
            if self.selftest["running"]:            # it does what the room says once the headphone test
                return                              # is done (_run_selftest)
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
            for probe in self.probes:
                probe.stop()
            self.prefs.pop("restore_output", None)
            self._save_prefs()          # with what the session learned about its outputs (lead, IO level)
            gc.unfreeze()               # what it left behind can go now (see _started)
            gc.collect()
        self.idle_since = time.monotonic()

    def _started(self, session, probes):
        self.session = session
        self.probes = tuple(probe.start() for probe in probes)
        # Everything that exists now stays while the music plays: Python's full garbage collections,
        # which hold every Python thread while they walk all objects, never walk it again.
        gc.freeze()

    def _testing(self) -> bool:
        """The headphone test has the output to itself for its 30 s: no sharing or listening meanwhile."""
        if self.selftest["running"]:
            self._fail(sa.SetupError("busy", "The headphone test is still running. Try again when it's done."))
        return self.selftest["running"]

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
            if self._testing():
                return False
            from_room = from_room or getattr(self.session, "from_room", False)
            self._stop_session()
            self.error = None
            self.firewall(fresh=True)       # asked now, before the music starts, not every few seconds
            probes = (ae.GcTimer(), ae.GilProbe())
            try:
                self.refresh_devices()
                session = sa.HostSession(self.sd, self.name, delay_ms=self.prefs.get("delay_ms"), click=click,
                                         play_local=self.prefs.get("play_local", True), settings=self.settings,
                                         port=self.port, switch_tool=self.switch_tool, log=self._log,
                                         devices=self.prefs["devices"], peer_leads=self.prefs["peer_leads"],
                                         probes=probes)
                session.start()
            except Exception as e:
                self._fail(e)
                return False
            session.from_room = from_room
            self._started(session, probes)
            if session.system and session.system.previous:
                self.prefs["restore_output"] = session.system.previous
                self._save_prefs()
            return True

    def join(self, address: str, port: int | None = None, from_room: bool = False) -> bool:
        address = (address or "").strip()
        if not address:
            return False
        with self.lock:
            if self._testing():
                return False
            self._stop_session()
            self.error = None
            probes = (ae.GcTimer(), ae.GilProbe())
            try:
                self.refresh_devices()      # the output macOS plays on now (AirPods put in since), not at launch
                session = sa.JoinSession(self.sd, address, self.name, settings=self.settings,
                                         port=port or self.port, switch_tool=self.switch_tool, log=self._log,
                                         devices=self.prefs["devices"], probes=probes)
                session.from_room = from_room
                session.start()
            except Exception as e:
                self._fail(e)
                return False
            self._started(session, probes)
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
            if self.session:
                self.session.event("offset_change", uid=s.device_key, ms=s.offset_ms)
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
            delay = (None if data["delay_ms"] is None
                     else min(sa.DelayPolicy.D_MAX * 1000, max(40.0, float(data["delay_ms"]))))
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
        """The macOS firewall's view of this app. Asking takes three processes, so it is asked when
        sharing starts and when the person asks (Allow SyncAudio, Copy diagnostics); in between, the
        answer is kept for 5 minutes."""
        if sys.platform != "darwin" or not os.path.exists(FIREWALL):
            return None
        if fresh or self._firewall is None or time.monotonic() - self._firewall_at > 300:
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

    def start_selftest(self, output: str | None = None):
        """"Test my headphones": a quiet tone for 10 s the way 1.x played it (may crackle), then 20 s
        through SyncAudio 2.0, and a verdict: clean, macOS dropping audio, or SyncAudio falling behind."""
        with self.lock:
            if self.selftest["running"]:
                return
            if self.session:
                raise sa.SetupError("busy", "Stop sharing or listening first: the test needs the headphones "
                                            "to itself for 30 seconds.")
            self.refresh_devices()          # headphones connected since the app opened
            sd = self.sd
            index = sa.pick_output(sd, output or None)
            test = ae.SelfTest(sd, index, log=self._log)
            self._selftest = test
            self.selftest = {"running": True, "output": sd.query_devices(index)["name"], "started": time.time(),
                             "seconds": list(test.seconds), "result": None}
        threading.Thread(target=self._run_selftest, args=(test,), name="selftest", daemon=True).start()

    def _run_selftest(self, test):
        try:
            result = test.run()
            self.selftest.update(result=result, message=sa.VERDICTS[result["verdict"]])
            self._log(f"headphone test on {result['output']}: {result['verdict']}")
        except Exception as e:
            self.selftest.update(message=f"The test couldn't play on this output: {e}")
            self._log(f"! headphone test: {e!r}")
        finally:
            self.selftest["running"] = False
        if self.room:           # anything the connected Macs chose meanwhile
            self._room_changed()

    def diagnostics(self) -> str:
        """Everything useful for working out what's wrong, as text to paste (spec 11)."""
        try:
            with sa.PORTAUDIO:      # (a session may be restarting PortAudio this very moment)
                devices = [device_facts(d) for d in self.sd.query_devices()]
        except Exception as e:
            devices = str(e)
        ip = sa.lan_ip()
        iface = discovery.interface_for(ip)
        firewall = self.firewall(fresh=True)
        session = self.session.info() if self.session else None
        audio = pings = None
        if session:     # these get their own place in the report
            audio = {"output": session.pop("audio")["output"],         # what each stream opened as (11.1)
                     "capture": (session.get("capture") or {}).pop("input", None)}
            pings = session.pop("clock_pings", None)
        report = {
            "app": self.build, "macos": platform.mac_ver()[0], "machine": platform.machine(),
            "system": system_info() if sys.platform == "darwin" else None,
            "network": {"ip": ip, "subnet": str(iface[0]) if iface else None,
                        "firewall": (firewall or {}).get("raw", {}).get("global"), "firewall_app": firewall,
                        "scan_send_errors": self.scan_errors, "known_hosts": self.prefs.get("known_hosts", []),
                        "nearby": self.nearby, "bonjour": bool(discovery.dns_sd()),
                        "wifi": wifi_info() if sys.platform == "darwin" else None},
            "devices": devices,
            "audio": audio,
            "session": session,
            "delay_events": [e for e in (session or {}).get("events", []) if e.get("event") in DELAY_EVENTS],
            "clock_pings": pings,
            "python": self.python_info(),
            "selftest": self.selftest,
            "room": self.room.status() if self.room else None,
            "settings": {**self.state()["settings"], "offset_ms_1x": self.prefs.get("offset_ms_1x")},
            "history": list(self.history),
            "events": list(self.events),
            "probe": self.probe_result,
            "coreaudiod_overloads": coreaudio_overloads() if sys.platform == "darwin" else None,
        }
        return json.dumps(report, indent=1, default=str)

    def python_info(self) -> dict:
        """Python's own stalls, which can hold up the thread that feeds the speakers: the longest garbage
        collections of the session, how often threads take turns, and which threads there are."""
        timer = next((p for p in self.probes if isinstance(p, ae.GcTimer)), None)
        return {"version": platform.python_version(),
                "switch_interval_ms": round(sys.getswitchinterval() * 1000, 3),
                "gc_top_ms": [round(t * 1000, 2) for t in reversed(timer.top)] if timer else None,
                "gc_gen2": timer.gen2 if timer else None, "gc_frozen": gc.get_freeze_count(),
                "threads": sorted(t.name for t in threading.enumerate())}

    def summary(self, report: dict) -> str:
        """The few numbers that matter, short enough to paste into a chat (spec 11.4)."""
        s, net = report.get("session") or {}, report["network"]
        wifi = net.get("wifi") or {}
        lines = [f"SyncAudio {report['app'].get('version')} ({(report['app'].get('commit') or 'dev')[:7]}) "
                 f"macOS {report['macos']} {(report.get('system') or {}).get('model') or ''}".rstrip(),
                 f"network {net['ip']} {net['subnet']} "
                 f"firewall {'on' if (net.get('firewall_app') or {}).get('enabled') else 'off'} "
                 f"wifi {wifi.get('channel')} {wifi.get('signal')}",
                 f"mode {s.get('mode', 'idle')} delay {s.get('delay_ms')} ms output {s.get('output')}"]
        history = report.get("history", [])
        changes = [e for e in report.get("delay_events", []) if e.get("event") == "delay_change"]
        glides = sum(e.get("mode") == "glide" for e in changes)
        delay = f"delay {s.get('delay_ms')} ms ({len(changes) - glides} steps, {glides} glides)"
        if s.get("mode") == "hosting":
            if s.get("local"):
                lines += player_summary("this Mac", s["local"], [(h["t"], h.get("local")) for h in history], delay)
            cap = s.get("capture")
            lines.append((f"capture {cap.get('name')} io {io_text(cap.get('cap_io'))} skips {cap.get('cap_skips')} "
                          f"({cap.get('cap_skipped_ms')} ms) overflows {cap.get('cap_overflows')} "
                          f"gap {cap.get('cap_gap_frames')} frames reader max {cap.get('cap_reader_max_ms')} ms "
                          f"dll {cap.get('cap_dll_ppm')} ppm" if cap else "capture: test clicks")
                         + f" | level {s.get('level')} silent {s.get('silent_s')} s")
            for l in s.get("listeners", []):
                if l.get("old"):
                    lines.append(f"listener {l.get('name')} has SyncAudio {l.get('version')}: it needs 2.0 to play")
                    continue
                past = [(h["t"], next((x for x in h.get("listeners", []) if x.get("id") == l.get("id")), None))
                        for h in history]
                label = (f"listener {l.get('name')} (v{l.get('version')}, {l.get('codec')}, "
                         f"wifi {l.get('wifi_ms')} ms, late {l.get('late')}, unsent {l.get('unsent')} B)")
                lines += player_summary(label, l, past, delay)
        elif s:
            label = (f"listening to {s.get('host_name')} (v{s.get('host_version')}, {s.get('connection')}, "
                     f"{s.get('codec')}, wifi {s.get('wifi_ms')} ms, silent {s.get('incoming_silent_s')} s)")
            lines += player_summary(label, s, [(h["t"], h) for h in history if h.get("mode") == "listening"], delay)
            if s.get("problem"):
                lines.append(f"problem {s['problem'].get('code')}: {s['problem'].get('message')}")
        if (report.get("selftest") or {}).get("result"):
            lines.append(selftest_summary(report["selftest"]["result"]))
        recent = history[-30:]
        if recent:
            keys = ("state", "err_ms", "headroom_ms", "dropouts", "delay_ms")
            def compact(h):
                src = h.get("local") or h
                return "/".join(str(src.get(k, h.get(k))) for k in keys)
            lines.append("last 30 s (state/err/headroom/dropouts/delay): " + " ".join(compact(h) for h in recent[::3]))
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
            self.history.append(self._sample(info))
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
            "selftest": {**self.selftest, "phase": getattr(self._selftest, "phase", None)},
        }

    @staticmethod
    def _sample(info: dict) -> dict:
        """One second of history (spec 11.2): everything the session's players reported."""
        sample = {"t": round(time.time(), 1), "mode": info["mode"], "delay_ms": info.get("delay_ms")}
        if info["mode"] == "hosting":
            sample.update({k: info.get(k) for k in ("delay_target_ms", "glide")}, local=info["local"],
                          listeners=[{k: l.get(k) for k in PEER_KEYS} for l in info["listeners"]],
                          **{k: v for k, v in (info.get("capture") or {}).items() if k.startswith("cap_")})
        else:
            sample.update({k: v for k, v in info.items() if k not in NOT_SAMPLED})
        return sample


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
                "/api/selftest": lambda: controller.start_selftest(data.get("output")),
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


def audio_core_problem(sounddevice) -> str | None:
    """What's wrong with the audio core (libsacore) or the sounddevice it is built around; None if
    nothing. In the app, libsacore must be the copy inside it: it's the only thing that can play."""
    print("sacore:", sacore.LIBRARY, hex(sacore.VERSION or 0))
    if sounddevice.__version__ != "0.5.6" or not hasattr(sounddevice, "_StreamBase"):
        return (f"sounddevice {sounddevice.__version__}: the audio engine hands PortAudio its C callback through "
                "sounddevice 0.5.6's _StreamBase")
    frozen = getattr(sys, "frozen", False)
    if not sacore.available():
        return "libsacore isn't in the app: it couldn't play anything" if frozen else None
    if frozen and sacore.LIBRARY != os.path.join(sys._MEIPASS, "libsacore.dylib"):
        return f"libsacore was loaded from {sacore.LIBRARY}, not from inside the app"
    if abs(sacore.now() - time.monotonic()) >= 1e-3:
        return "libsacore's clock isn't the one PortAudio's timestamps use"
    if not sacore.smoke_test():
        return "libsacore's output callback didn't play back what was written"
    return None


def self_test() -> int:
    """Used by the build: proves the packaged app has everything it needs."""
    import numpy
    import sounddevice
    print("numpy", numpy.__version__, "| sounddevice", sounddevice.__version__,
          "|", sounddevice.get_portaudio_version()[1])
    print("audio devices:", len(sounddevice.query_devices()))
    problem = audio_core_problem(sounddevice)
    if problem:
        print(problem)
        return 1
    try:
        import webview
        print("pywebview", getattr(webview, "__version__", "ok"))
    except ImportError as e:
        print("pywebview missing:", e)
        return 1
    print("SwitchAudioSource:", bundled_switch_tool() or "not bundled")
    print("Opus:", sa.codec.LIBRARY or "missing")
    if getattr(sys, "frozen", False) and not sa.codec.available():
        print("Opus is missing from the app: listeners would get uncompressed audio")
        return 1
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

    # No audio thread needs the GIL any more (CoreAudio's threads only run C); this is fairness for the
    # output engine's thread, which should get Python back within a millisecond when it wakes.
    sys.setswitchinterval(0.001)
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
            gc.freeze()     # the app as it starts stays: see Controller._started
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
            gc.freeze()
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
