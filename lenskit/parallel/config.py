# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Optional


def proc_count(core_div: Optional[int] = 2, max_default: Optional[int] = None, level: int = 0):
    """
    Get the number of desired jobs for multiprocessing operations.  This does not
    affect Numba or MKL multithreading.

    This count can come from a number of sources:

    * The ``LK_NUM_PROCS`` environment variable
    * The number of CPUs, divided by ``core_div`` (default 2)

    Args:
        core_div:
            The divisor to scale down the number of cores; ``None`` to turn off core-based
            fallback.
        max_default:
            The maximum number of processes to use if the environment variable is not
            configured.
        level:
            The process nesting level.  0 is the outermost level of parallelism; subsequent
            levels control nesting.  Levels deeper than 1 are rare, and it isn't expected
            that callers actually have an accurate idea of the threading nesting, just that
            they are configuring a child.  If the process count is unconfigured, then level
            1 will use ``core_div``, and deeper levels will use 1.

    Returns:
        int: The number of jobs desired.
    """

    nprocs = os.environ.get("LK_NUM_PROCS", None)
    if nprocs is not None:
        nprocs = [int(s) for s in nprocs.split(",")]
    elif core_div is not None:
        nprocs = max(mp.cpu_count() // core_div, 1)
        if max_default is not None:
            nprocs = min(nprocs, max_default)
        nprocs = [nprocs, core_div]
    assert isinstance(nprocs, list)

    if level >= len(nprocs):
        return 1
    else:
        return nprocs[level]
