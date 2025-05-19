# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Old home of the LensKit logging worker logic.

.. deprecated:: 2025.3
    This module is deprecated.  Import from :mod:`lenskit.logging.multiprocess`.
"""

from .multiprocess import WorkerContext, WorkerLogConfig, send_task

__all__ = ["WorkerContext", "WorkerLogConfig", "send_task"]
