# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from lenskit.data.types import NPVector


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
        return remap[res]  # type: ignore

    if n >= 0 and n < N:
        parts = np.argpartition(-xs, n)
        top_scores = xs[parts[:n]]
        top_sort = np.argsort(-top_scores)
        order = parts[top_sort]
    else:
        order = np.argsort(-xs)

    return order  # type: ignore
