# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import scipy.sparse as sps

from pytest import approx

from lenskit.data import ItemList
from lenskit.metrics import GeometricRankWeight, entropy, rank_biased_entropy


def test_entropy_basic():
    items = ItemList([1, 2, 3], ordered=True)
    categories = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert entropy(items, categories, n=3) == approx(np.log2(3))


def test_entropy_partial_and_k_dense():
    items = ItemList([1, 2, 3, 4], ordered=True)
    categories = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    assert entropy(items, categories, n=2) == approx(1.0)


def test_rbe_basic():
    items = ItemList([1, 2, 3], ordered=True)
    categories = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = rank_biased_entropy(items, categories, n=3)
    assert 0 < result <= np.log2(3)


def test_rbe_weight():
    items1 = ItemList([1, 2, 3, 4, 5], ordered=True)
    categories1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0]])
    items2 = ItemList([1, 2, 3, 4, 5], ordered=True)
    categories2 = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # is category 2 less diverese?
    assert rank_biased_entropy(items1, categories1, n=5) > rank_biased_entropy(
        items2, categories2, n=5
    )


def test_rbe_sparse_dense_equivalence():
    items = ItemList([1, 2, 3], ordered=True)
    dense = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sparse = sps.csr_matrix(dense)
    # is there consistency between dense and sparse matrices?
    assert rank_biased_entropy(items, dense, n=3) == approx(rank_biased_entropy(items, sparse, n=3))


def test_rbe_custom_weight():
    items = ItemList([1, 2, 3], ordered=True)
    categories = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    w = GeometricRankWeight(0.5)
    result = rank_biased_entropy(items, categories, weight=w, n=3)
    assert 0 < result <= np.log2(3)
