"""Records a loopback device's input for the Mac CI (tests/coreaudio), in its own unstressed process.

It captures the way 2.0 does - libsacore's C input callback (512-frame buffers, latency 0), so no
Python runs on the IO thread and nothing the process under test does can make the recording miss
- drains the ring every 50 ms, and on SIGTERM / SIGINT (or after --seconds) writes
    PREFIX.pcm.npy    every frame captured, by device frame index from 0
    PREFIX.recs.npy   the callbacks' log (sacore.LOG_DTYPE): ADC time of each buffer
    PREFIX.rec.json   the device's facts and the ring's counters
It prints READY once callbacks flow, so run_all.py starts what it records only then.

    python tests/coreaudio/recorder.py --device "BlackHole 2ch" --out ca-p3/bh
"""
import argparse
import json
import os
import signal
import sys
import threading
import time

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(here)))
import sounddevice as sd  # noqa: E402
import sacore  # noqa: E402

RATE = 48000


def find(name: str, kind: str) -> int:
    for i, d in enumerate(sd.query_devices()):
        if d["name"] == name and d[f"max_{kind}_channels"] >= 2:
            return i
    sys.exit(f"recorder: no {kind} device named {name!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", required=True)
    ap.add_argument("--out", required=True, help="prefix of the files written")
    ap.add_argument("--seconds", type=float, default=600.0, help="stop by itself after this long")
    args = ap.parse_args(argv)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    ring = sacore.Ring("input", 2, RATE)
    stream = sd._StreamBase(kind="input", samplerate=RATE, blocksize=512, device=find(args.device, "input"),
                            channels=2, dtype="float32", latency=0.0, callback=ring.callback,
                            userdata=sacore.userdata(sd, ring.h), wrap_callback=None)
    chunks, recs, pos, lapped = [], [], 0, 0

    def drain():
        nonlocal pos, lapped
        recs.append(ring.new_records())
        w = int(ring.state()["w"])
        if w > pos:
            try:
                chunks.append(ring.read(pos, w - pos))
            except sacore.Lapped:       # can't happen at 50 ms a drain, but must never go unnoticed
                lapped += 1
                chunks.append(np.zeros((w - pos, 2), np.float32))
            pos = w

    stream.start()
    began = time.monotonic()
    while sum(len(r) for r in recs) < 3:
        if time.monotonic() - began > 10:
            sys.exit(f"recorder: {args.device} never called back")
        recs.append(ring.new_records())
        time.sleep(0.01)
    print("READY", flush=True)
    while not stop.wait(0.05) and time.monotonic() - began < args.seconds:
        drain()
    drain()
    stream.abort()
    stream.close()
    drain()
    st = ring.state()
    ring.close()
    pcm = np.concatenate(chunks) if chunks else np.zeros((0, 2), np.float32)
    recs = np.concatenate(recs)
    np.save(args.out + ".pcm.npy", pcm)
    np.save(args.out + ".recs.npy", recs)
    meta = {"device": args.device, "rate": RATE, "frames": len(pcm), "callbacks": len(recs),
            "facts": sacore.mac_call("device_info", args.device, False),
            "skips": int(st["skips"]), "skipped_frames": int(st["skipped_frames"]),
            "overflows": int(st["overflows"]), "lapped": lapped, "flag_or": int(st["flag_or"]),
            "max_cb_us": round(st["max_cb_ns"] / 1000, 1), "fake": getattr(stream, "stats", None)}
    with open(args.out + ".rec.json", "w") as f:
        json.dump(meta, f, indent=1)
    peak = float(np.abs(pcm).max()) if len(pcm) else 0.0
    print(f"recorded {len(pcm) / RATE:.1f} s from {args.device}: peak {peak:.3f}, {meta['skips']} skipped cycles",
          flush=True)


if __name__ == "__main__":
    main()
