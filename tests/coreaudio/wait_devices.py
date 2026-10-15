"""Waits for the Mac CI's loopback devices after coreaudiod restarts, and checks they are what the
tests assume (tests/coreaudio).

PortAudio only lists the devices that existed when it started, so every look is a fresh process.
A device counts once two looks in a row agree on it. The facts asserted are the ones the judge
relies on (from CoreAudio, through sacore's macOS helper):
  BlackHole 2ch     latency, safety offset and stream latency all 0: what is played at sample time S
                    is recorded at S, so timing errors are absolute
  SyncTestBT 2ch    safety offset 3840 + stream latency 3840 = 160 ms of fixed latency, like AirPods
  SyncTestSpk 2ch   none (a second plain loopback)
A --need device missing or wrong fails (exit 1). The --want devices (the twins) are optional: if they
don't load, or load wrong, that is a ::warning:: and twins=no, so a broken fixture can't block a
release; stock BlackHole covers every gate. --json gets what it found, --github-output `twins=yes|no`.

    python tests/coreaudio/wait_devices.py --need "BlackHole 2ch" --want "SyncTestBT 2ch" "SyncTestSpk 2ch"
"""
import argparse
import json
import os
import subprocess
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(here))
FACTS = {"BlackHole 2ch": {"latency": 0, "safety": 0, "stream_latency": 0},
         "SyncTestBT 2ch": {"latency": 0, "safety": 3840, "stream_latency": 3840},
         "SyncTestSpk 2ch": {"latency": 0, "safety": 0, "stream_latency": 0}}
LOOK = """
import json, sounddevice as sd, sacore
out = []
for d in sd.query_devices():
    out.append({"name": d["name"], "inputs": d["max_input_channels"], "outputs": d["max_output_channels"],
                "out": sacore.mac_call("device_info", d["name"], True),
                "in": sacore.mac_call("device_info", d["name"], False)})
print(json.dumps(out))
"""


def look() -> dict:
    """{name: device} as a fresh PortAudio sees it, or {} if that failed (coreaudiod still starting)."""
    try:
        run = subprocess.run([sys.executable, "-c", LOOK], capture_output=True, text=True, timeout=20, cwd=root,
                             env={**os.environ, "PYTHONPATH": os.pathsep.join(
                                 p for p in (os.environ.get("PYTHONPATH"), root) if p)})
    except subprocess.TimeoutExpired:
        print("  (listing the devices took over 20 s)", file=sys.stderr)
        return {}
    if run.returncode:
        print("  (listing the devices failed: " + (run.stderr.strip().splitlines() or ["?"])[-1] + ")", file=sys.stderr)
        return {}
    return {d["name"]: d for d in json.loads(run.stdout)}


def wrong(name: str, dev: dict) -> list:
    """What isn't as the tests assume."""
    bad = [] if dev["inputs"] >= 2 and dev["outputs"] >= 2 else [f"{dev['inputs']} in / {dev['outputs']} out channels"]
    for side in ("out", "in"):
        facts = dev[side]
        if not facts:
            bad.append(f"no CoreAudio facts for its {side}put (sacore's macOS helper)")
            continue
        bad += [f"{side}put {k} {facts[k]}, not {v}" for k, v in FACTS[name].items() if facts[k] != v]
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--need", nargs="+", default=["BlackHole 2ch"])
    ap.add_argument("--want", nargs="*", default=[])
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--grace", type=float, default=15.0, help="s to keep waiting for --want once --need is there")
    ap.add_argument("--json", help="write what was found here")
    ap.add_argument("--github-output", help="append twins=yes|no to this file")
    args = ap.parse_args(argv)
    start = time.monotonic()
    last, seen, looks, ready_at = None, {}, 0, None
    while True:
        now = look()
        looks += 1
        stable = {k: v for k, v in now.items() if last and k in last}      # there two looks in a row
        last = now
        if stable:
            seen = stable
        have = [n for n in args.need if n in seen]
        if len(have) == len(args.need) and ready_at is None:
            ready_at = time.monotonic()
        waited = time.monotonic() - start
        if ready_at is not None and (all(n in seen for n in args.want) or time.monotonic() - ready_at >= args.grace):
            break
        if waited >= args.timeout:
            break
        time.sleep(1.0)
    report = {"looks": looks, "waited_s": round(time.monotonic() - start, 1),
              "devices": [seen[n] for n in sorted(seen)]}
    failed = [f"{n}: never appeared" for n in args.need if n not in seen]
    for n in args.need:
        problems = wrong(n, seen[n]) if n in seen and n in FACTS else []
        if problems:
            failed.append(f"{n}: {', '.join(problems)}")
    twins = "yes" if args.want else "no"
    for n in args.want:
        problem = "never appeared" if n not in seen else ", ".join(wrong(n, seen[n])) if n in FACTS else ""
        if problem:
            print(f"::warning::{n}: {problem}. Its phases are skipped (stock BlackHole still covers every gate).")
            twins = "no"
    report.update(twins=twins, failed=failed)
    print(f"after {report['waited_s']} s ({looks} looks): " + ", ".join(sorted(seen)) + f"; twins={twins}")
    for d in report["devices"]:
        f = d["out"] or {}
        print(f"  {d['name']}: {d['inputs']} in / {d['outputs']} out, latency {f.get('latency')} safety "
              f"{f.get('safety')} stream latency {f.get('stream_latency')} buffer {f.get('buffer_frames')} "
              f"[{f.get('buffer_min')}-{f.get('buffer_max')}] {f.get('nominal_rate')} Hz")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    if args.github_output:
        with open(args.github_output, "a") as f:
            f.write(f"twins={twins}\n")
    if failed:
        print("Loopback devices not as the tests need them:\n  " + "\n  ".join(failed), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
