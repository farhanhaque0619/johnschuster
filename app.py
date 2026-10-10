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
import hashlib
import http.server
import json
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

import syncaudio as sa

APP_NAME = "SyncAudio"
VERSION = "1.0.0"
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


def resource_dir() -> Path:
    """Where ui/ and the bundled helper live (inside the .app when packaged)."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def bundled_switch_tool() -> str | None:
    tool = resource_dir() / "SwitchAudioSource"
    return str(tool) if tool.exists() else None


def default_prefs_file() -> Path:
    return Path.home() / "Library" / "Application Support" / APP_NAME / "prefs.json"


class Controller:
    """Owns the current session (sharing or listening) and everything the UI shows."""

    def __init__(self, sd=None, prefs_file: Path | None = None, port: int = sa.PORT, scan: bool = True,
                 opener=None, switch_tool: str | None = None):
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
        self.error = None
        self.blackhole = None
        self.nearby: list = []
        self.idle_since = time.monotonic()
        self.install = {"state": "idle", "message": ""}
        self.events: deque = deque(maxlen=40)
        self.lock = threading.RLock()
        self.running = True
        self._recover_sound_output()
        if scan:
            threading.Thread(target=self._scan_loop, daemon=True).start()

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
        try:
            self.prefs_file.parent.mkdir(parents=True, exist_ok=True)
            self.prefs_file.write_text(json.dumps(self.prefs, indent=2))
        except OSError as e:
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
        while self.running:
            if self.session is None:
                found = sa.discover_all(self.port, timeout=1.0)
                if self.session is None:
                    self.nearby = [{"address": ip, "name": name} for ip, name in found]
            time.sleep(2.0)

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

    def share(self, click: bool = False):
        with self.lock:
            self._stop_session()
            self.error = None
            try:
                self.refresh_devices()
                session = sa.HostSession(self.sd, self.name, delay_ms=self.prefs.get("delay_ms"), click=click,
                                         play_local=self.prefs.get("play_local", True), settings=self.settings,
                                         port=self.port, switch_tool=self.switch_tool, log=self._log)
                session.start()
            except Exception as e:
                return self._fail(e)
            self.session = session
            if session.system and session.system.previous:
                self.prefs["restore_output"] = session.system.previous
                self._save_prefs()

    def join(self, address: str):
        address = (address or "").strip()
        if not address:
            return
        with self.lock:
            self._stop_session()
            self.error = None
            try:
                self.session = sa.JoinSession(self.sd, address, self.name, settings=self.settings,
                                              port=self.port, switch_tool=self.switch_tool, log=self._log).start()
            except Exception as e:
                return self._fail(e)
            self.prefs["last_host"] = address
            self._save_prefs()

    def stop(self):
        with self.lock:
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

    def copy(self, text: str):
        try:
            subprocess.run(["pbcopy"], input=str(text)[:500], text=True, check=False, timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass

    def shutdown(self):
        self.running = False
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
        previous = getattr(getattr(session, "system", None), "previous", None)
        if isinstance(session, sa.HostSession) and previous and self.prefs.get("restore_output") != previous:
            self.prefs["restore_output"] = previous
            self._save_prefs()
        s = self.settings
        return {
            "app": {"name": APP_NAME, "version": VERSION, "download_url": DOWNLOAD_URL},
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
                "/api/stop": controller.stop,
                "/api/settings": lambda: controller.update(data),
                "/api/install-blackhole": controller.install_blackhole,
                "/api/check-devices": controller.refresh_devices,
                "/api/open": lambda: controller.open(str(data.get("what"))),
                "/api/copy": lambda: controller.copy(str(data.get("text", ""))),
                "/api/dismiss": lambda: setattr(controller, "error", None),
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
    args, _ = ap.parse_known_args(argv)     # Finder may add its own arguments
    if args.self_test:
        sys.exit(self_test())

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
