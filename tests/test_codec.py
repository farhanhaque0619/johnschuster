"""Opus compression of the stream, and reading the Wi-Fi band for diagnostics."""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app  # noqa: E402
import codec  # noqa: E402


@unittest.skipUnless(codec.available(), "libopus isn't installed here (the app bundles it)")
class OpusTest(unittest.TestCase):
    def test_a_tenth_of_the_traffic_and_a_known_delay(self):
        rate = codec.RATE
        t = np.arange(rate) / rate
        music = np.stack([0.5 * np.sin(2 * np.pi * 440 * t), 0.3 * np.sin(2 * np.pi * 660 * t)], 1).astype(np.float32)
        music[20000] = 0.9                                  # a click to time
        enc, dec = codec.Encoder(2), codec.Decoder(2)
        packets = [enc.encode(music[i:i + codec.FRAME]) for i in range(0, rate, codec.FRAME)]
        sent = sum(len(p) for p in packets)
        self.assertLess(sent, rate * 2 * 2 / 8, "an eighth of 16-bit stereo or less")
        out = np.concatenate([dec.decode(p) for p in packets])
        self.assertEqual(out.shape, music.shape)
        lag = int(np.argmax(np.correlate(out[:30000, 0], music[:30000, 0], "full"))) - 29999
        self.assertEqual(lag, codec.DELAY, "the delay the listener takes out")
        diff = out[codec.DELAY:] - music[:-codec.DELAY]
        snr = 10 * np.log10((music[2000:40000] ** 2).sum() / (diff[2000:40000] ** 2).sum())
        self.assertGreater(snr, 25)

    def test_odd_sized_blocks_are_not_encoded(self):
        self.assertIsNone(codec.Encoder(2).encode(np.zeros((441, 2), np.float32)))


class WifiTest(unittest.TestCase):
    def profile(self, channel):
        return json.dumps({"SPAirPortDataType": [{"spairport_airport_interfaces": [{
            "_name": "en0", "spairport_current_network_information": {
                "_name": "<redacted>", "spairport_network_channel": channel,
                "spairport_network_phymode": "802.11ax", "spairport_signal_noise": "-58 dBm / -94 dBm"}}]}]})

    def test_reads_the_band(self):
        self.assertEqual(app.wifi_info(self.profile("149 (5GHz, 80MHz)"))["band"], "5 GHz")
        self.assertEqual(app.wifi_info(self.profile("6 (2GHz, 20MHz)"))["band"], "2.4 GHz")
        self.assertEqual(app.wifi_info(self.profile("11"))["band"], "2.4 GHz")
        self.assertEqual(app.wifi_info(self.profile("37 (6GHz, 160MHz)"))["signal"], "-58 dBm / -94 dBm")
        self.assertIsNone(app.wifi_info("{}"))
        self.assertIsNone(app.wifi_info("not json"))


class OverloadLogTest(unittest.TestCase):
    def test_counts_macos_overload_lines(self):
        log = ("Timestamp               Ty Process[PID:TID]\n"
               "2026-10-14 03:01:02.123 E  coreaudiod[180:2b1] [com.apple.coreaudio:as_server] HALS_IOContext: "
               "skipping cycle due to overload\n"
               "2026-10-14 03:01:02.130 Df coreaudiod[180:2b1] something else\n"
               "2026-10-14 03:01:03.001 E  coreaudiod[180:2b1] IOWorkLoop: Overload detected\n")
        lines = app.coreaudio_overloads(log)
        self.assertEqual(lines[0], "2 in the last 15 min")
        self.assertEqual(len(lines), 3)
        self.assertEqual(app.coreaudio_overloads(log, limit=1)[1:], [lines[-1]])


if __name__ == "__main__":
    unittest.main()
