# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Small utility functions or classes that don't fit elsewhere, mostly used inside
LensKit implementations.
"""

from ._indent import IndentWriter
from ._latch import Latch

__all__ = ["Latch", "IndentWriter"]
