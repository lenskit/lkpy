# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Literal

import numpy as np
import pyarrow as pa
import scipy.sparse as sps

from lenskit.data import ItemList, Vocabulary
from lenskit.data.attributes import attr_set
from lenskit.data.schema import AttrLayout, ColumnSpec

from ._base import ListMetric, RankingMetricBase
from ._weighting import RankWeight


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

    n = min(n, len(items))

    truncated = categories[:n, :]

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

    n = min(n, len(items))

    truncated = categories[:n, :]

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

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.nan

    if weights is not None:
        matrix = matrix * weights[:, np.newaxis]

    values = np.asarray(matrix.sum(axis=0))
    values = values + 1e-6

    total = np.sum(values)
    probs = values / total
    entropy = float(-np.sum(probs * np.log2(probs)))

    return entropy


class Entropy(ListMetric, RankingMetricBase):
    """
    Evaluate diversity using Shannon entropy over item categories.

    Args:
        field: Name of the field containing categories
        layout: ('scalar', 'list', 'vector', 'sparse')
        n: Maximum recommendation list length
        normalize: Normalization method for the category matrix
    """

    def __init__(
        self,
        field: str,
        layout: Literal["scalar", "list", "vector", "sparse"],
        n: int | None = None,
        *,
        normalize: Literal["unit", "distribution"] | None = None,
    ):
        super().__init__(n)
        self.field = field
        self.layout = AttrLayout[layout.upper()]
        self.normalize = normalize

    @property
    def label(self):
        if self.n is not None:
            return f"Entropy@{self.n}"
        else:
            return "Entropy"

    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        recs = self.truncate(recs)

        field_data = recs.field(self.field, format="arrow")

        # build arrow table with this field
        table = pa.Table.from_arrays([field_data], names=[self.field])

        # get a vocabulary from the item list
        vocab = recs.vocabulary
        if vocab is None:
            vocab = Vocabulary(recs.ids(format="arrow"))

        # create the attributeset and call call_matrix
        spec = ColumnSpec(layout=self.layout)
        attrset = attr_set(self.field, spec, table, vocab, None)
        matrix, cat_vocab = attrset.cat_matrix(normalize=self.normalize)

        return entropy(recs, matrix, n=self.n)
