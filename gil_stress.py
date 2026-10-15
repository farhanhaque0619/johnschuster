"""Deliberate Python stress, to prove the audio never depends on Python being quick.

Only `syncaudio --stress {app,ci}` and the tests import this. Since 2.0 no audio thread needs the
GIL, so none of this may ever be heard; under 1.1.6 the same load was static.

  app  a bad day in the app itself: big json dumps, numpy fits, an HMAC and a subprocess every
       2 s, short pure-Python bursts
  ci   far worse, for CI: two threads that never stop computing, a full garbage collection of a
       300k-object heap every second, a 60k-entry json dump and a fork twice a second, numpy fits,
       and a C call that holds the GIL for 100 ms every 5 s
"""
from __future__ import annotations

import ctypes
import gc
import hashlib
import hmac
import json
import os
import subprocess
import threading
import time

import numpy as np

TRUE = "/usr/bin/true"


class Stress:
    """Stress(kind).start() ... .stop(). The threads are daemons; stop() lets them go."""

    def __init__(self, kind: str = "ci"):
        if kind not in ("app", "ci"):
            raise ValueError(f"unknown stress {kind!r}")
        self.kind = kind
        self.done = threading.Event()
        self.threads: list = []
        self.heap = None

    def start(self):
        for job in (self._ci() if self.kind == "ci" else self._app()):
            t = threading.Thread(target=job, name=f"stress-{self.kind}", daemon=True)
            t.start()
            self.threads.append(t)
        return self

    def stop(self):
        self.done.set()
        for t in self.threads:
            t.join(2)
        self.heap = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def _every(self, period: float, fn):
        def loop():
            while not self.done.wait(period):
                fn()
        return loop

    def _busy(self):
        x = 0
        while not self.done.is_set():
            for i in range(2000):
                x += i * i % 7

    @staticmethod
    def _fit():
        x = np.linspace(0, 1, 2000)
        np.polyfit(x, np.sin(x) + 0.01 * x * x, 3)

    @staticmethod
    def _spawn():
        subprocess.run([TRUE], check=False)

    def _ci(self):
        self.heap = [{"k": i, "v": [i, str(i)]} for i in range(300_000)]   # what a full gc has to walk
        big = {str(i): [i, i * 0.5, "x" * 10] for i in range(60_000)}
        hold = ctypes.PyDLL(None).usleep            # PyDLL: the GIL stays held for the whole call
        return [self._busy, self._busy,
                self._every(1.0, gc.collect),
                self._every(0.5, lambda: json.dumps(big)),
                self._every(0.5, self._spawn),
                self._every(0.2, self._fit),
                self._every(5.0, lambda: hold(100_000))]

    def _app(self):
        doc = {f"key{i}": {"n": i, "s": "value " * 3} for i in range(3000)}
        key, data = os.urandom(32), os.urandom(1 << 20)

        def burst():                                # 1-2 ms of pure Python
            end = time.perf_counter() + 0.001 + 0.001 * (time.perf_counter() % 1)
            while time.perf_counter() < end:
                pass
        return [self._every(0.5, lambda: json.dumps(doc)),
                self._every(0.2, self._fit),
                self._every(2.0, lambda: hmac.new(key, data, hashlib.sha256).digest()),
                self._every(2.0, self._spawn),
                self._every(0.05, burst)]
