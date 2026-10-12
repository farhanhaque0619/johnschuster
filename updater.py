"""
Self-update for the packaged SyncAudio.app.

The build bakes build_info.json (version + commit) into the app and publishes
SyncAudio.dmg on the "app-latest" GitHub release. To update, the app downloads
that .dmg with curl (so macOS doesn't quarantine it and ask "Open Anyway"
again), copies the new app out of it, quits, and a tiny shell script swaps the
bundles and opens the new one.
"""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "farhanhaque0619/johnschuster"
RELEASE_API = f"https://api.github.com/repos/{REPO}/releases/tags/app-latest"
APP = "SyncAudio.app"


def build_info(resources: Path, fallback_version: str) -> dict:
    try:
        return json.loads((resources / "build_info.json").read_text())
    except (OSError, ValueError):
        return {"version": fallback_version, "commit": None}


def _run(cmd: list, timeout: float = 120) -> str:
    done = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        raise RuntimeError((done.stderr or done.stdout).strip() or f"{cmd[0]} failed")
    return done.stdout


def latest_release() -> dict | None:
    """{"version", "commit", "dmg_url"} of the newest build, or None."""
    try:
        data = json.loads(_run(["curl", "-fsSL", "--max-time", "15", "-H", "Accept: application/vnd.github+json",
                                RELEASE_API], timeout=20))
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError):
        return None
    dmg = next((a["browser_download_url"] for a in data.get("assets", []) if a.get("name") == "SyncAudio.dmg"), None)
    if not dmg:
        return None
    return {"version": (data.get("name") or "").split()[-1] if data.get("name") else None,
            "commit": data.get("target_commitish"), "dmg_url": dmg}


def current_bundle() -> Path | None:
    """The running SyncAudio.app, if this is the packaged app and it can be replaced."""
    if not getattr(sys, "frozen", False):
        return None
    bundle = Path(sys.executable).resolve().parents[2]
    return bundle if bundle.suffix == ".app" else None


def staging_dir(bundle: Path) -> Path:
    """A scratch folder next to the app, so the final swap is a same-disk rename."""
    if "AppTranslocation" in str(bundle):
        raise RuntimeError("drag SyncAudio into your Applications folder first, then open it from there")
    try:
        return Path(tempfile.mkdtemp(prefix=".SyncAudio-update-", dir=bundle.parent))
    except OSError:
        raise RuntimeError(f"no permission to change {bundle.parent}")


def prepare(source: str, staging: Path) -> Path:
    """Fetch the .dmg (URL or local path) and copy the new app out of it."""
    dmg = staging / "SyncAudio.dmg"
    if source.startswith("https://"):
        _run(["curl", "-fsSL", "--retry", "2", "-o", dmg, source], timeout=900)
    else:
        shutil.copy(source, dmg)
    mount = staging / "mount"
    mount.mkdir()
    _run(["hdiutil", "attach", "-nobrowse", "-readonly", "-noautoopen", "-mountpoint", mount, dmg])
    try:
        if not (mount / APP).is_dir():
            raise RuntimeError("the download doesn't contain SyncAudio")
        new = staging / APP
        _run(["ditto", mount / APP, new])
    finally:
        subprocess.run(["hdiutil", "detach", "-force", str(mount)], capture_output=True)
    _run(["codesign", "--verify", "--deep", "--strict", new])
    return new


def swap_script(old: Path, new: Path, staging: Path, pid: int, relaunch: str | None) -> str:
    """Shell script: wait for the app to quit, put the new one in its place, reopen it.
    If anything goes wrong the old app is put back."""
    o, n, st = shlex.quote(str(old)), shlex.quote(str(new)), shlex.quote(str(staging))
    backup = shlex.quote(str(staging / "previous.app"))
    return "\n".join([
        f"for i in $(seq 1 150); do kill -0 {pid} 2>/dev/null || break; sleep 0.2; done",
        f"if mv {o} {backup} && mv {n} {o}; then rm -rf {st}; else [ -d {o} ] || mv {backup} {o}; fi",
        relaunch or "true",
    ])


def apply(new: Path, bundle: Path, staging: Path, pid: int, relaunch: bool = True) -> subprocess.Popen:
    """Start the swap in the background; the caller should quit right after."""
    script = swap_script(bundle, new, staging, pid, f"open {shlex.quote(str(bundle))}" if relaunch else None)
    return subprocess.Popen(["/bin/sh", "-c", script], start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
