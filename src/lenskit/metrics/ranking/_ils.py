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


def intra_list_similarity(items: ItemList, categories: np.ndarray | sps.spmatrix) -> float:
    """
    Compute intra-list similarity between item category vectors.

    Args:
        items: Item list to evaluate.
        categories: Item * category matrix (dense or sparse), unit-normalized.

    Returns:
        Average pairwise cosine similarity or NaN if insufficient data.
    """

    n = categories.shape[0]

    if len(items) == 0 or n == 0:
        return np.nan

    if n <= 1:
        return 1.0

    if sps.issparse(categories):
        similarity_matrix = np.array((categories @ categories.T).toarray())
    else:
        similarity_matrix = np.array(categories @ categories.T)

    ils_score = np.sum(np.triu(similarity_matrix, 1)) / (n * (n - 1) / 2)

    return float(ils_score)


class ILS(ListMetric, RankingMetricBase):
    """
    Evaluate recommendation diversity using intra-list similarity (ILS).

    This metric measures the average pairwise cosine similarity between item
    category vectors in a recommendation list. Lower values indicate more diverse
    recommendations (items from different categories), while higher values indicate
    less diverse recommendations (items from similar categories).

    Args:
        dataset: The LensKit dataset containing item entities and their attributes.
        attribute: Name of the attribute to use for categories (e.g., 'genre', 'tag').
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

        categories = self._cat_matrix[item_nums, :]

        return intra_list_similarity(recs, categories)
