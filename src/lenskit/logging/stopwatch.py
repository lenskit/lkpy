# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Timing support
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from .formats import friendly_duration


class Stopwatch:
    """
    Timer class for recording elapsed wall time in operations.
    """

    acc_time: float | None = None
    start_time: float | None = None
    stop_time: float | None = None

    def __init__(self, start=True):
        if start:
            self.start()

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.stop_time = time.perf_counter()
        if self.start_time is not None and self.acc_time is not None:
            self.acc_time += self.stop_time - self.start_time

    def elapsed(self, *, accumulated=True) -> float:
        """
        Get the elapsed time.
        """
        assert self.start_time is not None
        stop = self.stop_time or time.perf_counter()

        t = stop - self.start_time
        if accumulated and self.acc_time is not None:
            t += self.acc_time

        return t

    @contextmanager
    def measure(self, accumulate: bool = False):
        """
        Context manager to measure an item, optionally accumulating its time.
        """
        if accumulate:
            if self.acc_time is None:
                self.acc_time = 0.0
        else:
            self.acc_time = None

        self.start()
        try:
            yield self
        finally:
            self.stop()

    def __str__(self):
        elapsed = self.elapsed()
        return friendly_duration(elapsed)

    def __repr__(self):
        elapsed = self.elapsed()
        if self.stop_time:
            return "<Stopwatch stopped at {:.3f}s>".format(elapsed)
        else:
            return "<Stopwatch running at {:.3f}s>".format(elapsed)
