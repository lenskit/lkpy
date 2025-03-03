# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Measure resource consumption.
"""

# pyright: strict
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ._proxy import get_logger

_log = get_logger(__name__)


@dataclass
class ResourceMeasurement:
    """
    Single measurement of resources.  Two measurements can be subtracted to
    compute the time resources consumed in an interval (memory resources are
    left unchanged).

    Stability:
        Internal
    """

    wall_time: float
    perf_time: float
    user_time: float
    system_time: float

    max_rss: int | None = None
    """
    Maximum RSS usage (in bytes).
    """
    max_gpu: int | None = None

    @classmethod
    def current(cls):
        """
        Get the current resource measurements.
        """
        wall = time.time()
        perf = time.perf_counter()
        rss = None
        gpu = None
        try:
            import resource

            ru = resource.getrusage(resource.RUSAGE_SELF)
            user = ru.ru_utime
            system = ru.ru_stime
            rss = ru.ru_maxrss * 1024
        except ImportError:
            ts = os.times()
            user = ts.user
            system = ts.system

        if torch.cuda.is_available():
            gpu = torch.cuda.max_memory_allocated()

        return cls(wall, perf, user, system, rss, gpu)

    @property
    def cpu_time(self) -> float:
        "Total CPU time (user + system)."
        return self.user_time + self.system_time

    def __sub__(self, other: ResourceMeasurement):
        return ResourceMeasurement(
            self.wall_time - other.wall_time,
            self.perf_time - other.perf_time,
            self.user_time - other.user_time,
            self.system_time - other.system_time,
            self.max_rss,
        )


def reset_linux_hwm():
    pid = os.getpid()
    reset_file = Path(f"/proc/{pid}/clear_refs")
    if reset_file.exists():
        try:
            reset_file.write_text("5")
        except IOError:
            _log.warn("cannot clear refs", pid=pid)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def max_memory():
    "Get the maximum memory use for this process"
    try:
        import resource
    except ImportError:
        return "unknown"
    else:
        res = resource.getrusage(resource.RUSAGE_SELF)
        return "%.1f MiB" % (res.ru_maxrss / 1024,)


def cur_memory():
    "Get the current memory use for this process"
    try:
        import resource
    except ImportError:
        return "unknown"
    else:
        res = resource.getrusage(resource.RUSAGE_SELF)
        return "%.1f MiB" % (res.ru_idrss / 1024,)
