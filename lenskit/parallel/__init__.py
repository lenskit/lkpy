# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit parallel computation support.
"""

from .config import proc_count
from .invoker import ModelOpInvoker, invoker

__all__ = [
    "invoker",
    "ModelOpInvoker",
    "proc_count",
]
