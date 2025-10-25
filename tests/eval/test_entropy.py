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
from lenskit.metrics.ranking._entropy import matrix_column_entropy


def test_entropy_uniform():
    items = ItemList([1, 2, 3], ordered=True)
    # dense matrix with uniform distribution
    dense = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # sparse matrix
    row = [0, 1, 2]
    col = [0, 1, 2]
    data = [1, 1, 1]
    sparse = sps.csr_matrix((data, (row, col)), shape=(3, 3))

    dense_result = entropy(items, dense)  # n is lenght of items i.e. 3
    sparse_result = entropy(items, sparse, n=5)  # n is max(5, len(items)) i.e. 3

    assert dense_result == approx(np.log2(3), abs=0.02)
    assert sparse_result == approx(dense_result, abs=0.001)


def test_entropy_partial():
    items = ItemList([1, 2, 3, 4], ordered=True)
    dense = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    row = [0, 1, 2, 3]
    col = [0, 1, 0, 1]
    data = [1, 1, 1, 1]
    sparse = sps.csr_matrix((data, (row, col)), shape=(4, 2))

    dense_result = entropy(items, dense, n=2)
    sparse_result = entropy(items, sparse, n=2)

    assert dense_result == approx(1.0, abs=0.02)
    assert sparse_result == approx(dense_result, abs=0.001)


def test_rank_biased_entropy():
    items = ItemList([1, 2, 3, 4, 5], ordered=True)
    # more diverse: 3 categories spread early
    row1 = [0, 1, 2, 3, 4]
    col1 = [0, 1, 2, 0, 0]
    data1 = [1, 1, 1, 1, 1]
    categories1 = sps.csr_matrix((data1, (row1, col1)), shape=(5, 3))
    # less diverse: category 0 dominates early
    row2 = [0, 1, 2, 3, 4]
    col2 = [0, 0, 0, 1, 2]
    data2 = [1, 1, 1, 1, 1]
    categories2 = sps.csr_matrix((data2, (row2, col2)), shape=(5, 3))

    result1 = rank_biased_entropy(items, categories1)  # n is length of items i.e. 5
    result2 = rank_biased_entropy(items, categories2, n=5)

    # categories1 more diverse due to weights
    assert 0 < result1 <= np.log2(3)
    assert 0 < result2 <= np.log2(3)
    assert result1 > result2


def test_matrix_column_entropy_weighted():
    dense = np.array([[1, 0], [0, 1], [1, 0]])

    row = [0, 1, 2]
    col = [0, 1, 0]
    data = [1, 1, 1]
    sparse = sps.csr_matrix((data, (row, col)), shape=(3, 2))
    weights = np.array([1.0, 0.5, 0.5])

    dense_unweighted = matrix_column_entropy(dense)
    dense_weighted = matrix_column_entropy(dense, weights=weights)
    sparse_unweighted = matrix_column_entropy(sparse)
    sparse_weighted = matrix_column_entropy(sparse, weights=weights)

    # weighted entropy differs from unweighted
    assert dense_weighted != dense_unweighted
    assert sparse_weighted != sparse_unweighted

    assert dense_unweighted == approx(sparse_unweighted, abs=0.001)
    assert dense_weighted == approx(sparse_weighted, abs=0.001)


def test_matrix_column_entropy_empty():
    dense_empty = np.array([]).reshape(0, 3)
    sparse_empty = sps.csr_matrix((0, 3))

    assert np.isnan(matrix_column_entropy(dense_empty))
    assert np.isnan(matrix_column_entropy(sparse_empty))


def test_matrix_column_entropy_all_zeros():
    dense_zeros = np.zeros((3, 4))
    sparse_zeros = sps.csr_matrix((3, 4))

    dense_result = matrix_column_entropy(dense_zeros)
    sparse_result = matrix_column_entropy(sparse_zeros)

    # uniform distribution due to smoothing
    assert dense_result == approx(np.log2(4), abs=0.02)
    assert sparse_result == approx(dense_result, abs=0.001)


def test_entropy_large_sparse():
    items = ItemList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ordered=True)
    # 10 items, 5 categories, sparse with 5 non-zero entries
    row = [0, 2, 4, 6, 8]
    col = [0, 1, 2, 3, 4]
    data = [1, 1, 1, 1, 1]
    categories = sps.csr_matrix((data, (row, col)), shape=(10, 5))

    result = entropy(items, categories, n=10)
    # uniform distribution across 5 categories
    assert result == approx(np.log2(5), abs=0.02)


def test_rank_biased_entropy_partial_weighted():
    items = ItemList([1, 2, 3, 4, 5], ordered=True)
    # Sparse matrix: 5 items, 3 categories
    row = [0, 1, 3]
    col = [0, 1, 2]
    data = [1, 1, 1]
    categories = sps.csr_matrix((data, (row, col)), shape=(5, 3))
    weight = GeometricRankWeight(0.5)

    result = rank_biased_entropy(items, categories, weight=weight, n=3)
    # entropy positive but less than log2(3) due to weights
    assert 0 < result <= np.log2(3)
