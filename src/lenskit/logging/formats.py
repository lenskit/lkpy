# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Formatting utilities for log and diagnostic output.
"""

from datetime import timedelta


def friendly_duration(elapsed: float | timedelta):
    """
    Short, human-friendly representation of a duration.
    """
    if isinstance(elapsed, timedelta):
        elapsed = elapsed.total_seconds()

    if elapsed < 1:
        return "{: 0.0f}ms".format(elapsed * 1000)
    elif elapsed > 60 * 60:
        h, m = divmod(elapsed, 60 * 60)
        m, s = divmod(m, 60)
        return "{:0.0f}h{:0.0f}m{:0.2f}s".format(h, m, s)
    elif elapsed > 60:
        m, s = divmod(elapsed, 60)
        return "{:0.0f}m{:0.2f}s".format(m, s)
    else:
        return "{:0.2f}s".format(elapsed)
