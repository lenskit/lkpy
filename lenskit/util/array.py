# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Array utilities.
"""

from numba import njit


@njit
def swap(a, i, j):
    t = a[i]
    a[i] = a[j]
    a[j] = t
