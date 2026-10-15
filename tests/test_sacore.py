"""sacore.py, the Python side of libsacore: which library it loads, and the guard around the
optional macOS helpers (a failure there must never reach the audio)."""
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
sys.path.insert(0, root)
import sacore  # noqa: E402


def loaded(env=None, meipass=None):
    """sacore.LIBRARY as a fresh Python process sees it."""
    code = (f"import sys; sys._MEIPASS = {meipass!r}; " if meipass else "") + \
        "import sacore; print(sacore.LIBRARY, sacore.available())"
    env = {k: v for k, v in os.environ.items() if k != "SACORE_LIB"} | (env or {})
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.split()


class LoaderTest(unittest.TestCase):
    def setUp(self):
        self.built = sacore.LIBRARY
        if not self.built:
            self.skipTest("libsacore isn't built")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_sacore_lib_wins_and_is_the_only_choice(self):
        self.assertEqual(loaded({"SACORE_LIB": self.built}), [self.built, "True"])
        self.assertEqual(loaded({"SACORE_LIB": os.path.join(self.tmp, "nothing.so")}), ["None", "False"])

    def test_the_app_bundle_comes_before_the_source_folder(self):
        bundled = os.path.join(self.tmp, "libsacore.dylib")
        shutil.copy(self.built, bundled)
        self.assertEqual(loaded(meipass=self.tmp), [bundled, "True"])

    def test_the_source_folder_build(self):
        name = "libsacore.dylib" if sys.platform == "darwin" else "libsacore.so"
        if not os.path.exists(os.path.join(root, name)):
            self.skipTest(f"no {name} next to sacore.py")
        self.assertEqual(loaded(), [os.path.join(root, name), "True"])


class BindingTest(unittest.TestCase):
    def test_userdata_is_a_cffi_pointer_for_sounddevice(self):
        sd = types.SimpleNamespace(_ffi=types.SimpleNamespace(cast=lambda t, h: ("cast", t, h)))
        self.assertEqual(sacore.userdata(sd, 1234), ("cast", "void *", 1234))
        self.assertEqual(sacore.userdata(object(), 1234), 1234)        # the tests' fake

    @unittest.skipUnless(sacore.available(), "libsacore isn't built")
    def test_bad_shapes_are_refused_before_reaching_c(self):
        with self.assertRaises(ValueError):
            sacore.Ring("output", 9, 48000)                             # more channels than the C allows
        ring = sacore.Ring("output", 2, 48000)
        self.addCleanup(ring.close)
        with self.assertRaises(ValueError):
            ring.write(10000, [[0.0, 0.0, 0.0]], 0)
        self.assertEqual(ring.write(10000, [[0.5, -0.5]], 0), 1)       # anything array-like
        ring.close()
        ring.close()                                                    # twice is fine
        content = sacore.Content(2, 48000)
        content.close()
        for use in (ring.state, ring.new_records, lambda: ring.write(0, [[0.0, 0.0]], 0), lambda: ring.cut(0, 0),
                    lambda: ring.read(0, 1), content.range, lambda: content.put(0, [[0.0, 0.0]])):
            with self.assertRaises(sacore.Closed):                      # not a NULL pointer handed to C
                use()


    @unittest.skipUnless(sacore.available() and shutil.which("nm"), "needs libsacore and nm")
    def test_libsacore_never_calls_into_python(self):
        # Nothing in it can take the GIL, so nothing on an IO thread can wait for Python.
        cmd = ["nm", "-u", sacore.LIBRARY] if sys.platform == "darwin" else ["nm", "-D", "-u", sacore.LIBRARY]
        undefined = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        self.assertIn("memcpy", undefined)
        self.assertNotRegex(undefined, r"\b_?Py")


class MacCallTest(unittest.TestCase):
    def test_without_the_macos_layer_everything_is_the_default(self):
        with mock.patch.object(sacore, "mac", None):
            self.assertIsNone(sacore.mac_call("device_info", "AirPods", True))
            self.assertEqual(sacore.mac_call("default_output", default=0), 0)
        if sys.platform != "darwin":
            self.assertIsNone(sacore.mac)

    def test_a_failing_helper_is_reported_once_and_never_raises(self):
        def broken(*_):
            raise OSError("CoreAudio said no")
        fake = types.SimpleNamespace(device_info=broken, default_output=lambda: 73)
        err = io.StringIO()
        with mock.patch.object(sacore, "mac", fake), mock.patch.object(sacore, "_mac_failed", set()), \
                contextlib.redirect_stderr(err):
            self.assertEqual(sacore.mac_call("device_info", "AirPods", True, default={}), {})
            self.assertEqual(sacore.mac_call("device_info", "AirPods", True, default={}), {})
            self.assertEqual(sacore.mac_call("default_output"), 73)
            self.assertIsNone(sacore.mac_call("set_nominal_rate", 73, 44100.0))  # an older library
        self.assertEqual(err.getvalue().count("device_info"), 1)


if __name__ == "__main__":
    unittest.main()
