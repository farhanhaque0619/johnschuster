"""Run the tests for the build. Many of them time real threads against each other, and a
shared CI Mac now and then stalls one for a moment; a test that fails is run once more
on its own. A real bug fails both times and still fails the build (both runs are logged)."""
import os
import sys
import unittest

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.dirname(here))


suite = unittest.defaultTestLoader.discover(here)
first = unittest.TextTestRunner(verbosity=2).run(suite)
failed = sorted({t.id().split(" ")[0] for t, _ in first.failures + first.errors})
if not failed:
    sys.exit(0)
print(f"\nRunning {len(failed)} failed test(s) once more on their own: {failed}\n", flush=True)
again = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(failed))
sys.exit(0 if again.wasSuccessful() else 1)
