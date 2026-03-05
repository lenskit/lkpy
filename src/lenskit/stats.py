# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit statistical computations.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

from lenskit.data.types import NPVector
from lenskit.diagnostics import DataWarning


def gini(xs: ArrayLike) -> float:
    """
    Compute the Gini coefficient of an array of values.

    This is inspired by `Olivia Guest's implementation`_ and based on the
    `StatsDirect reference`_.  It does *not* include Olivia's zero adjustment;
    the Gini coefficient is fine with some zeros, so long as the sum is strictly
    positive.

    .. _Olivia Guest's implementation: https://github.com/oliviaguest/gini
    .. _StatsDirect reference: https://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    Stability:
        Caller

    Args:
        xs:
            An array of values representing the “resource” allocated to each
            instance.

    Returns:
        The Gini coefficient of the values in ``xs``.
    """
    xs = np.asarray(xs)
    if np.any(xs < 0):
        warnings.warn(
            "Gini coefficient is not defined for negative values", DataWarning, stacklevel=2
        )

    n = len(xs)
    xs = np.sort(xs)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    ranks *= 2
    ranks -= n + 1
    num = np.sum(xs * ranks)
    denom = n * np.sum(xs, dtype=np.float64)
    if denom <= 0:
        warnings.warn(
            "Gini coefficient is not defined for non-positive totals", DataWarning, stacklevel=2
        )
    return max(num / denom, 0)


def argtopn(xs: ArrayLike, n: int) -> NPVector[np.int64]:
    """
    Compute the ordered positions of the top *n* elements.  Similar to
    :func:`torch.topk`, but works with NumPy arrays and only returns the
    indices.

    .. deprecated:: 2025.3.0

        This was never declared stable, but is now deprecated and will be
        removed in 2026.1.
    """
    if n == 0:
        return np.empty(0, np.int64)

    xs = np.asarray(xs)

    N = len(xs)
    invalid = np.isnan(xs)
    if np.any(invalid):
        mask = ~invalid
        vxs = xs[mask]
        remap = np.arange(N)[mask]
        res = argtopn(vxs, n)
        return remap[res]

    if n >= 0 and n < N:
        parts = np.argpartition(-xs, n)
        top_scores = xs[parts[:n]]
        top_sort = np.argsort(-top_scores)
        order = parts[top_sort]
    else:
        order = np.argsort(-xs)

    return order
