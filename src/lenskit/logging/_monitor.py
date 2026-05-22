# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Old home of the LensKit logging monitor.

.. deprecated:: 2025.3
    This module is deprecated.  Import from :mod:`lenskit.logging.multiprocess`.
"""

from .multiprocess import Monitor, get_monitor

__all__ = ["Monitor", "get_monitor"]
