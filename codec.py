"""
Opus compression for the audio stream.

Uncompressed CD-quality stereo is 1.5 Mbit/s. On a Mac with Bluetooth
headphones that much Wi-Fi traffic is a problem: Wi-Fi and Bluetooth share one
radio, and the headphones lose audio while Wi-Fi is busy (static). Opus brings
the stream to about a tenth of that, with no audible difference.

libopus is bundled with the app; when it can't be loaded (or the sample rate
isn't one Opus supports), the stream simply stays uncompressed.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import sys

import numpy as np

RATE = 48000
FRAME = 480                 # 10 ms: one capture block
LOWDELAY = 2051             # OPUS_APPLICATION_RESTRICTED_LOWDELAY: music codec, 2.5 ms delay
DELAY = 120                 # decoded audio comes out this many samples late (2.5 ms at 48 kHz)
MAX_PACKET = 4000


def _candidates():
    base = getattr(sys, "_MEIPASS", None)               # inside the packaged app
    if base:
        yield from sorted(glob.glob(os.path.join(base, "libopus*.dylib")))
    found = ctypes.util.find_library("opus")
    if found:
        yield found
    yield from ("/opt/homebrew/lib/libopus.0.dylib", "/usr/local/lib/libopus.0.dylib", "libopus.so.0")


def _load():
    for path in _candidates():
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        err = ctypes.POINTER(ctypes.c_int)
        lib.opus_encoder_create.restype = ctypes.c_void_p
        lib.opus_encoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, err]
        lib.opus_encode_float.restype = ctypes.c_int
        lib.opus_encode_float.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_void_p, ctypes.c_int]
        lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
        lib.opus_decoder_create.restype = ctypes.c_void_p
        lib.opus_decoder_create.argtypes = [ctypes.c_int, ctypes.c_int, err]
        lib.opus_decode_float.restype = ctypes.c_int
        lib.opus_decode_float.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                          ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
        return lib, path
    return None, None


_lib, LIBRARY = _load()


def available(rate: int = RATE, channels: int = 2) -> bool:
    return _lib is not None and rate == RATE and channels in (1, 2)


class Encoder:
    def __init__(self, channels: int):
        err = ctypes.c_int()
        self.channels = channels
        self._enc = _lib.opus_encoder_create(RATE, channels, LOWDELAY, ctypes.byref(err))
        if err.value or not self._enc:
            raise RuntimeError(f"opus encoder: error {err.value}")
        self._out = ctypes.create_string_buffer(MAX_PACKET)

    def encode(self, pcm: np.ndarray) -> bytes | None:
        """One 10 ms block (480 x channels float32) -> an Opus packet (None if it can't)."""
        if pcm.shape != (FRAME, self.channels):
            return None
        pcm = np.ascontiguousarray(pcm, np.float32)
        n = _lib.opus_encode_float(self._enc, pcm.ctypes.data, FRAME, self._out, MAX_PACKET)
        return self._out.raw[:n] if n > 0 else None

    def __del__(self):
        if getattr(self, "_enc", None):
            _lib.opus_encoder_destroy(self._enc)


class Decoder:
    def __init__(self, channels: int):
        err = ctypes.c_int()
        self.channels = channels
        self._dec = _lib.opus_decoder_create(RATE, channels, ctypes.byref(err))
        if err.value or not self._dec:
            raise RuntimeError(f"opus decoder: error {err.value}")
        self._pcm = np.zeros((FRAME * 6, channels), np.float32)     # room for the longest packet

    def decode(self, packet: bytes) -> np.ndarray:
        n = _lib.opus_decode_float(self._dec, packet, len(packet), self._pcm.ctypes.data, FRAME * 6, 0)
        if n < 0:
            raise ValueError(f"opus decode: error {n}")
        return self._pcm[:n].copy()

    def __del__(self):
        if getattr(self, "_dec", None):
            _lib.opus_decoder_destroy(self._dec)
