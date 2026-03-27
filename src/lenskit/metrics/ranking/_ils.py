# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_array

from lenskit.data import Dataset, ItemList, Vocabulary

from ._base import ListMetric, RankingMetricBase


def intra_list_similarity(items: ItemList, vectors: np.ndarray | sps.spmatrix) -> float:
    """
    Compute intra-list similarity between item vectors.

    Args:
        items: Item list to evaluate.
        vectors: Matrix (dense or sparse) where each row is a
        unit-normalized vector representing an item.

    Returns:
        Average pairwise cosine similarity or NaN if insufficient data.
    """

    n = vectors.shape[0]

    if len(items) == 0 or n == 0:
        return np.nan

    if n <= 1:
        return 1.0

    if sps.issparse(vectors):
        similarity_matrix = np.array((vectors @ vectors.T).toarray())
    else:
        similarity_matrix = np.array(vectors @ vectors.T)

    ils_score = np.sum(np.triu(similarity_matrix, 1)) / (n * (n - 1) / 2)

    return float(ils_score)


class ILS(ListMetric, RankingMetricBase):
    """
    Evaluate recommendation diversity using intra-list similarity (ILS).

    This metric measures the average pairwise cosine similarity between item
    vectors in a recommendation list. Lower values indicate more diverse
    recommendations, while higher values indicate less diverse recommendations.

    Args:
        dataset: The LensKit dataset containing item entities and their attributes.
        attribute: Name of the attribute or vector source (e.g., 'genre', 'tag').
        n: Recommendation list length to evaluate.

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

        # compute category matrix
        self._cat_matrix, _ = attr_set.cat_matrix(normalize="unit")
        self._item_vocab = items.vocabulary

    @property
    def label(self):
        base = f"ILS({self.attribute})"
        if self.n is not None:
            return f"{base}@{self.n}"
        return base

    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        recs = self.truncate(recs)

        item_nums = recs.numbers(vocabulary=self._item_vocab, missing="negative")
        item_nums = item_nums[item_nums >= 0]
        if len(item_nums) == 0:
            return np.nan

        vectors = self._cat_matrix[item_nums, :]

        return intra_list_similarity(recs, vectors)
