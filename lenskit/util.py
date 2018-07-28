"""
Miscellaneous utility functions.
"""

import time


class Stopwatch():
    start_time = None
    stop_time = None

    def __init__(self, start=True):
        if start:
            self.start()

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.stop_time = time.perf_counter()

    def elapsed(self):
        stop = self.stop_time
        if stop is None:
            stop = time.perf_counter()

        return stop - self.start_time

    def __str__(self):
        elapsed = self.elapsed()
        if elapsed < 1:
            return "{: 0.0f}ms".format(elapsed * 1000)
        else:
            return "{: 0.2f}s".format(elapsed)
