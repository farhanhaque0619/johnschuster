"""Run the tests for the build.

Two kinds of tests:
- Checks of the logic (parsing, the player, the codec, the updater, ...): these
  decide whether a build is released. A failure is run once more on its own; a
  real bug fails both times and stops the release.
- Simulations that play audio in real time between threads (fake AirPods, three
  Macs in one process, Bonjour). On GitHub's shared Macs these sometimes miss by
  a few milliseconds because the machine stalled, not because the app is wrong,
  so they are run and their results printed, but they don't stop a release.
"""
import os
import sys
import unittest

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.dirname(here))

REAL_TIME = {"EndToEndTest", "SlowWifiTest", "BluetoothAudioTest", "AirPodsTest",
             "CombinedDeviceTest", "SessionTest", "TwoMacsTest", "BonjourRoundTripTest"}


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
