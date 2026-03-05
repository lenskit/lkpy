# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit parallel computation support.
"""

from __future__ import annotations

from .config import (
    effective_cpu_count,
    ensure_parallel_init,
    get_parallel_config,
    init_threading,
    is_free_threaded,
)

__all__ = [
    "init_threading",
    "get_parallel_config",
    "effective_cpu_count",
    "ensure_parallel_init",
    "is_free_threaded",
]
