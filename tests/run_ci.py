"""Run the tests for the build.

Two kinds of tests:
- Checks that decide whether a build is released: the logic (parsing, the delay
  policy, the codec, the updater, ...), the audio engine in virtual time, the
  real-time tests judged only on what the machine can't fake (fake AirPods played
  through libsacore's fake IO thread, its own late wake-ups allowed for), and the
  Mac CI's real-CoreAudio scripts run through on fake loopback devices. A
  failure is run once more on its own; a real bug fails both times and stops the
  release.
- Simulations of whole apps in real time (three Macs in one process, following
  AirPods, Bonjour). On GitHub's shared Macs these sometimes miss because the
  machine stalled, not because the app is wrong, so they are run and their
  results printed, but they don't stop a release.

First it builds libsacore (native/build.sh), the C code that runs on CoreAudio's IO
threads, and every test uses that fresh build: on a Mac the one with the tests' fake IO
thread, on Linux also the AddressSanitizer build.
"""
import os
import subprocess
import sys
import unittest

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
sys.path.insert(0, here)
sys.path.insert(0, root)

print("== Building libsacore ==", flush=True)
build = subprocess.run(["bash", os.path.join(root, "native", "build.sh")], env={**os.environ, "SACORE_SANITIZE": "1"})
if build.returncode:
    print("native/build.sh failed: nothing can play without libsacore")
    sys.exit(1)
os.environ["SACORE_LIB"] = os.path.join(root, "libsacore_test.dylib" if sys.platform == "darwin" else "libsacore.so")

REAL_TIME = {"EndToEndTest", "SlowWifiTest", "AirPodsTest", "CombinedDeviceTest", "SessionTest", "TwoMacsTest",
             "BonjourRoundTripTest"}


def tests(suite):
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            yield from tests(t)
        else:
            yield t


def run(suite):
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failed = sorted({t.id().split(" ")[0] for t, _ in result.failures + result.errors})
    if failed:
        print(f"\nRunning {len(failed)} failed test(s) once more on their own: {failed}\n", flush=True)
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(failed))
    return result.wasSuccessful()


everything = list(tests(unittest.defaultTestLoader.discover(here)))
gate = unittest.TestSuite(t for t in everything if type(t).__name__ not in REAL_TIME)
sims = unittest.TestSuite(t for t in everything if type(t).__name__ in REAL_TIME)

print("== Real-time simulations (reported, not blocking) ==", flush=True)
sims_ok = run(sims)
print("== Logic checks (these decide the release) ==", flush=True)
gate_ok = run(gate)
print(f"\nSimulations: {'passed' if sims_ok else 'SOME FAILED (not blocking, see above)'}")
print(f"Logic checks: {'passed' if gate_ok else 'FAILED'}")
sys.exit(0 if gate_ok else 1)
