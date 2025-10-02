# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT


from __future__ import annotations

import numpy as np

from lenskit.data import ItemList
from lenskit.metrics import GeometricRankWeight


def RBO(reference: ItemList, reranked: ItemList, p: float = 0.85, k: int = 10) -> float:
    """
    Computes the RBO metric defined in:
    Webber, William, Alistair Moffat, and Justin Zobel. "A similarity measure for indefinite rankings."
    ACM Transactions on Information Systems (TOIS) 28.4 (2010): 20.
    https://dl.acm.org/doi/10.1145/1852102.1852106

    Args:
        reference:
            The first item list to compare.
        reranked:
            The second item list to compare.
        p:
            The patience parameter (default 0.85).
        k:
            The depth to which to compute the overlap (default 10).

    Returns:
        The RBO score between 0 and 1.
    """
    rank_weight = GeometricRankWeight(p)
    weights = rank_weight.weight(np.arange(1, k + 1))
    sum = 0
    total_weights = 0
    for d, weight in enumerate(weights, start=1):
        set_a = set(reference[:d].ids())
        set_b = set(reranked[:d].ids())
        overlap = len(set_a.intersection(set_b))
        agreement = overlap / d
        sum += agreement * weight
        total_weights += weight
    rbo = sum / total_weights
    return rbo
