# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_array

from lenskit.data import Dataset, ItemList, Vocabulary

from ._base import ListMetric, RankingMetricBase
from ._weighting import GeometricRankWeight, RankWeight


def entropy(items: ItemList, categories: np.ndarray | sps.spmatrix) -> float:
    """
    Compute Shannon entropy over categorical distributions.

    Args:
        items: Item list to evaluate.
        categories: Item * category matrix (dense or sparse).

    Returns:
        Shannon entropy or NaN if no valid data is available.
    """
    if len(items) == 0:
        return np.nan

    return matrix_column_entropy(categories)


def rank_biased_entropy(
    items: ItemList, categories: np.ndarray | sps.spmatrix, *, weight: RankWeight | None = None
) -> float:
    """
    Compute rank-biased Shannon entropy over categorical distributions.

    Args:
        items: Item list to evaluate.
        categories: Item * category matrix (dense or sparse).
        weight: Optional RankWeight. Defaults to GeometricRankWeight(0.85).

    Returns:
        Rank-biased Shannon entropy or NaN if no valid data is available.
    """
    if len(items) == 0:
        return np.nan

    if weight is None:
        weight = GeometricRankWeight(0.85)

    ranks = np.arange(1, len(items) + 1)
    wvec = weight.weight(ranks)
    return matrix_column_entropy(categories, weights=wvec)


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

    This metric measures the diversity of categories in recommendation list.
    Higher entropy indicates more diverse category distribution.

    Args:
        dataset: The LensKit dataset containing item entities and their attributes.
        attribute: Name of the attribute to use for categories (e.g., 'genre', 'tag')
        n: Recommendation list length to evaluate

    Stability:
        Caller
    """

    attribute: str
    _cat_matrix: np.ndarray | csr_array
    _item_vocab: Vocabulary

    def __init__(
        self,
        dataset: Dataset,
        attribute: str,
        n: int | None = None,
    ):
        super().__init__(n)
        self.attribute = attribute

        # get items entity set / attribute set
        items = dataset.entities("item")
        attr_set = items.attribute(attribute)

        # compute category matrix once at initialization
        self._cat_matrix, _ = attr_set.cat_matrix(normalize="distribution")
        self._item_vocab = items.vocabulary

    @property
    def label(self):
        base = f"Entropy({self.attribute})"
        if self.n is not None:
            return f"{base}@{self.n}"
        return base

    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        recs = self.truncate(recs)

        item_nums = recs.numbers(vocabulary=self._item_vocab, missing="negative")
        item_nums = item_nums[item_nums >= 0]

        categories = self._cat_matrix[item_nums, :]

        return entropy(recs, categories)


class RankBiasedEntropy(ListMetric, RankingMetricBase):
    """
    Evaluate diversity using rank-biased Shannon entropy over item categories.

    This metric measures the diversity of categories in recommendation list
    with rank-based weighting, giving more importance to items at the top
    of the recommendation list.

    Args:
        dataset: The LensKit dataset containing item entities and their attributes.
        attribute: Name of the attribute to use for categories (e.g., 'genre', 'tag')
        n: Recommendation list length to evaluate
        weight: Rank weighting model. Defaults to GeometricRankWeight(0.85)

    Stability:
        Caller
    """

    attribute: str
    weight: RankWeight
    _cat_matrix: np.ndarray | csr_array
    _item_vocab: Vocabulary

    def __init__(
        self,
        dataset: Dataset,
        attribute: str,
        n: int | None = None,
        *,
        weight: RankWeight | None = None,
    ):
        super().__init__(n)
        self.attribute = attribute
        self.weight = weight if weight is not None else GeometricRankWeight(0.85)

        items = dataset.entities("item")
        attr_set = items.attribute(attribute)

        self._cat_matrix, _ = attr_set.cat_matrix(normalize="distribution")
        self._item_vocab = items.vocabulary

    @property
    def label(self):
        base = f"RBEntropy({self.attribute})"
        if self.n is not None:
            return f"{base}@{self.n}"
        return base

    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        recs = self.truncate(recs)

        item_nums = recs.numbers(vocabulary=self._item_vocab, missing="negative")
        item_nums = item_nums[item_nums >= 0]

        categories = self._cat_matrix[item_nums, :]

        return rank_biased_entropy(recs, categories, weight=self.weight)
