"""Plays judge.content() on a real output for the Mac CI (tests/coreaudio), the way an engine does.

    --engine v2         2.0: audio_engine.AudioOutput (libsacore's C callback, the producer thread),
                        acting on its watchdog as the app's sessions do
    --engine legacy116  SyncAudio 1.1.6's SoundDeviceOutput + Player (tests/legacy), its Python
                        callback and PortAudio's choice of buffer: --legacy-latency (1.1.6: 0.1) and
                        --legacy-block (1.1.7: 2048)

Sample n is stamped play_at(n) = T0 + n / 48000 + 0.5 and handed over 10 ms at a time as it would
be captured. --stress ci loads this process as gil_stress does the app's. --stall-at T makes the C
callback T s in spin --stall-us once (a forced overload); --rate-at T sets the device to 44.1 kHz
there and back 5 s later. Writes PREFIX.json: what was opened, the C counters second by second,
every callback macOS skipped cycles before, and the watchdog's actions.

    python tests/coreaudio/engine_play.py --engine v2 --device "BlackHole 2ch" --seconds 60 --stress ci --out ca-p3/v2
"""
import argparse
import ctypes
import importlib.util
import json
import os
import sys
import threading
import time
import types
from dataclasses import asdict

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(here))
sys.path.insert(0, root)
sys.path.insert(0, here)
import sounddevice as sd  # noqa: E402
import sacore  # noqa: E402
import audio_engine as ae  # noqa: E402
import gil_stress  # noqa: E402
from judge import RATE, content  # noqa: E402

D = 0.5
BLOCK = 480
SETTINGS = types.SimpleNamespace(offset_ms=0.0, volume=1.0, channel="both")


def log(msg):
    print(f"{time.monotonic():.3f} {msg}", file=sys.stderr, flush=True)


def find(name: str) -> int:
    for i, d in enumerate(sd.query_devices()):
        if d["name"] == name and d["max_output_channels"] >= 2:
            return i
    sys.exit(f"engine_play: no output named {name!r}")


class Feeder(threading.Thread):
    """The host: each 10 ms block as soon as it would have been captured, stamped T0 + n/rate + D."""

    def __init__(self, feed, T0):
        super().__init__(daemon=True)
        self.feed, self.T0 = feed, T0
        self.done = threading.Event()

    def run(self):
        n = 0
        while not self.done.is_set():
            wait = self.T0 + (n + BLOCK) / RATE - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.feed(n, self.T0 + n / RATE + D, content(n, BLOCK))
            n += BLOCK


class Engine(ae.OutputEngine):
    """The app's output engine, also noting each callback macOS skipped cycles before, and any
    callback that itself took 10 ms or more (the forced stall)."""

    noted: list = []

    def _callbacks(self, recs):
        hit = recs[(recs["skipped"] > 0) | (recs["exit_time"] - recs["entry_time"] >= 0.010)]
        for r in hit.tolist():
            self.noted.append({"entry": r[4], "dac": r[2], "pos": r[1], "frames": r[6],
                               "skipped_ms": round(r[8] / RATE * 1000, 2), "cb_ms": round((r[5] - r[4]) * 1000, 2)})
        super()._callbacks(recs)


def pa_device(stream):
    """The CoreAudio device PortAudio's own PaMacCore_GetStreamOutputDevice says the stream plays to."""
    try:
        get = ctypes.CDLL(sd._libname).PaMacCore_GetStreamOutputDevice
    except (OSError, AttributeError, TypeError):
        return None
    get.restype, get.argtypes = ctypes.c_uint32, [ctypes.c_void_p]
    return get(int(sd._ffi.cast("uintptr_t", stream._ptr))) or None


def play_v2(args, index, out_info):
    ae.OutputEngine = Engine
    player = ae.LocalPlayer(RATE, 2)

    def open_output(level=None, reason="start"):
        out = ae.AudioOutput(sd, index, RATE, player, SETTINGS, log=log, reason=reason, level=level)
        out_info["outputs"].append({"t": time.monotonic(), "reason": reason, "level": out.plan.level,
                                    "plan": asdict(out.plan), "dev_id": out.dev_id, "pa_device": pa_device(out.stream),
                                    "facts": out.facts, "frames": out.engine.clock.frames_max})
        return out
    out = open_output()
    started = time.monotonic()
    feeder = Feeder(player.sb.feed, started)
    feeder.start()
    timeline, actions = [], []
    stall_at = started + args.stall_at if args.stall_at is not None else None
    rate_at = started + args.rate_at if args.rate_at is not None else None
    tick = started
    try:
        while time.monotonic() < started + args.seconds:
            tick += 1.0
            time.sleep(max(0.0, tick - time.monotonic()))
            now = time.monotonic()
            if stall_at is not None and now >= stall_at:
                out.ring.debug_stall(args.stall_us)
                out_info["stall"] = {"t": time.monotonic(), "us": args.stall_us}
                stall_at = None
            if rate_at is not None and now >= rate_at:
                to = 44100 if "rate_changes" not in out_info else RATE
                ok = sacore.mac_call("set_nominal_rate", out.dev_id, to)
                out_info.setdefault("rate_changes", []).append({"t": time.monotonic(), "hz": to, "ok": bool(ok)})
                rate_at = rate_at + 5.0 if to != RATE else None
            timeline.append({"t": round(now - started, 3), **out.second()})
            act = out.watchdog()
            if act:
                actions.append({"t": round(now - started, 3), **act})
                log(f"watchdog: {act}")
                if act["action"] in ("reopen", "follow"):
                    out.close()
                    out = open_output(act.get("level"), act["reason"])
        timeline.append({"t": round(time.monotonic() - started, 3), **out.second()})
    finally:
        feeder.done.set()
        clock = out.engine.clock.stats()
        out.close()
        player.sb.close()
    f = out.facts or {}
    return {"T0": started, "started": started, "strlat": f["stream_latency"] / f["nominal_rate"]
            if f.get("nominal_rate") else 0.0, "io": [clock["frames_min"], clock["frames_max"]],
            "clock": clock, "timeline": timeline, "actions": actions, "noted": Engine.noted,
            "fake": getattr(out.stream, "stats", None)}


def legacy116():
    path = os.path.join(root, "tests", "legacy", "syncaudio116.py")
    spec = importlib.util.spec_from_file_location("syncaudio116", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Tap:
    """sounddevice as 1.1.6 sees it: OutputStream gets --legacy-block, and each callback's frames,
    DAC time and status flags are noted (into arrays made beforehand: no allocation on the way)."""

    N = 1 << 21

    def __init__(self, block):
        self.block = block
        self.frames, self.dac, self.flags = np.zeros(self.N, np.int64), np.zeros(self.N), np.zeros(self.N, np.int64)
        self.k = 0
        self.stream = None

    def query_devices(self, *a, **kw):
        return sd.query_devices(*a, **kw)

    def OutputStream(self, callback, **kw):
        def noted(outdata, frames, t, status):
            k = self.k
            if k < self.N:
                self.frames[k], self.dac[k] = frames, t.outputBufferDacTime
                self.flags[k] = getattr(status, "_flags", 0)
                self.k = k + 1
            callback(outdata, frames, t, status)
        self.stream = sd.OutputStream(callback=noted, blocksize=self.block, **kw)
        return self.stream

    def numbers(self, skip_s=0.5) -> dict:
        f, d = self.frames[:self.k], self.dac[:self.k]
        first = int(np.searchsorted(np.cumsum(f), int(skip_s * RATE)))
        ok = (d[first + 1:] > 0) & (d[first:-1] > 0)
        gaps = (np.diff(d[first:]) - f[first:-1] / RATE)[ok]
        lost = gaps[gaps > 0.5 * f[first:-1][ok] / RATE].sum() if len(gaps) else 0.0
        return {"io": [int(f.min()), int(f.max())] if self.k else None, "callbacks": self.k,
                "cb_lost_ms": round(float(lost) * 1000, 1),
                "pa_flags": int(np.bitwise_or.reduce(self.flags[:self.k])) if self.k else 0}


def play_legacy(args, index, out_info):
    legacy = legacy116()
    legacy.OUTPUT_LATENCY = args.legacy_latency
    tap = Tap(args.legacy_block)
    player = legacy.Player(RATE, 2, lambda t: t)
    facts = sacore.mac_call("device_info", sd.query_devices(index)["name"], True)
    sel = getattr(sacore.mac, "SEL", None)
    counters = [c for c in (sacore.mac_call("listen", facts["id"], sel["overload"], sel[scope])
                            for scope in ("global", "wildcard")) if c] if facts and sel else []
    out = legacy.SoundDeviceOutput(tap, index, RATE, player.render)
    started = time.monotonic()
    feeder = Feeder(player.feed, started)
    feeder.start()
    try:
        time.sleep(args.seconds)
    finally:
        feeder.done.set()
        out.close()
    overloads = max((c.value for c in counters), default=0)
    for c in counters:
        c.close()
    out_info["outputs"].append({"t": started, "reason": "legacy", "facts": facts,
                                "latency": args.legacy_latency, "blocksize": args.legacy_block})
    return {"T0": started, "started": started, "strlat": facts["stream_latency"] / facts["nominal_rate"]
            if facts and facts.get("nominal_rate") else 0.0, "overloads": overloads, **tap.numbers(),
            "underflows": out.underflows, "fake": getattr(tap.stream, "stats", None)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", choices=["v2", "legacy116"], default="v2")
    ap.add_argument("--device", required=True)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--stress", choices=["none", "ci"], default="none")
    ap.add_argument("--legacy-latency", type=float, default=0.1)
    ap.add_argument("--legacy-block", type=int, default=0)
    ap.add_argument("--stall-at", type=float)
    ap.add_argument("--stall-us", type=int, default=60000)
    ap.add_argument("--rate-at", type=float)
    ap.add_argument("--out", required=True, help="prefix of the stats file (PREFIX.json)")
    args = ap.parse_args(argv)
    sys.setswitchinterval(0.001)            # as the app: fairness for the producer thread
    index = find(args.device)
    info = {"engine": args.engine, "device": args.device, "rate": RATE, "D": D, "seconds": args.seconds,
            "stress": args.stress, "clock_delta": abs(sacore.now() - time.monotonic()) if sacore.lib else None,
            "outputs": []}
    stress = gil_stress.Stress("ci").start() if args.stress == "ci" else None
    cpu, wall = time.process_time(), time.monotonic()
    try:
        info.update((play_v2 if args.engine == "v2" else play_legacy)(args, index, info))
    finally:
        if stress:
            stress.stop()
    info["stopped"] = time.monotonic()
    info["cpu_pct"] = round((time.process_time() - cpu) / (info["stopped"] - wall) * 100, 2)     # of one core
    with open(args.out + ".json", "w") as f:
        json.dump(info, f, indent=1, default=str)
    log(f"played {args.seconds:.0f} s on {args.device} ({args.engine}): io {info.get('io')}")


if __name__ == "__main__":
    main()
