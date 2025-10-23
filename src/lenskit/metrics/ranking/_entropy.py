# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from lenskit.data import ItemList


def entropy(
    items: ItemList, categories: np.ndarray | sps.spmatrix, *, n: int | None = None
) -> float:
    """
    Compute Shannon entropy over categorical distributions.

    Args:
        items: Item list to evaluate.
        categories: Item * category matrix (dense or sparse).
        n: Optional depth to evaluate; defaults to full list.

    Returns:
        Shannon entropy or NaN if no valid data is available.
    """
    if n is None:
        n = len(items)
    if n == 0:
        return np.nan

    truncated = categories[:n]
    return matrix_column_entropy(truncated)


def rank_biased_entropy(
    items: ItemList, categories: np.ndarray | sps.spmatrix, *, weight=None, n: int | None = None
) -> float:
    """
    Compute rank-biased Shannon entropy over categorical distributions.

    Args:
        items: Item list to evaluate.
        categories: Item * category matrix (dense or sparse).
        weight: Optional RankWeight. Defaults to GeometricRankWeight(0.85).
        n: Optional depth to evaluate; defaults to full list.

    Returns:
        Rank-biased Shannon entropy or NaN if no valid data is available.
    """
    from lenskit.metrics import GeometricRankWeight

    if n is None:
        n = len(items)
    if n == 0:
        return np.nan

    truncated = categories[:n]

    if weight is None:
        weight = GeometricRankWeight(0.85)

    ranks = np.arange(1, n + 1)
    wvec = weight.weight(ranks)
    return matrix_column_entropy(truncated, weights=wvec)


def matrix_column_entropy(
    matrix: np.ndarray | sps.spmatrix, weights: np.ndarray | None = None
) -> float:
    """
    Compute Shannon entropy from a matrix of items * categories.

    Args:
        matrix: Dense or sparse array of item * category values.
        weights: Optional per-item weight vector. If None, all items are equal.

    Returns:
        Shannon entropy (float) or np.nan if matrix is empty or all zeros.
    """
    if sps.issparse(matrix):
        matrix = matrix.toarray()

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.nan

    if weights is not None:
        matrix = matrix * weights[:, np.newaxis]

    counts = np.sum(matrix, axis=0)
    counts = counts[counts > 0]

    if len(counts) == 0 or np.sum(counts) == 0:
        return np.nan

    probs = counts / np.sum(counts)
    return float(-np.sum(probs * np.log2(probs)))
