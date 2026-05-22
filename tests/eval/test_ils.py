# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import scipy.sparse as sps

from pytest import approx

from lenskit.data import ItemList
from lenskit.metrics import ILS
from lenskit.metrics.ranking._ils import intra_list_similarity


def test_ils_identical_items():
    items = ItemList([1, 2, 3], ordered=True)
    # all items have same category vector
    dense = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])

    result = intra_list_similarity(items, dense)
    assert result == approx(1.0, abs=0.001)


def test_ils_orthogonal_items():
    items = ItemList([1, 2, 3], ordered=True)
    # orthogonal unit vectors
    dense = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    result = intra_list_similarity(items, dense)
    assert result == approx(0.0, abs=0.001)


def test_ils_sparse_identical():
    items = ItemList([1, 2, 3], ordered=True)
    # sparse matrix
    row = [0, 1, 2]
    col = [0, 0, 0]
    data = [1, 1, 1]
    sparse = sps.csr_array((data, (row, col)), shape=(3, 3))

    result = intra_list_similarity(items, sparse)
    assert result == approx(1.0, abs=0.001)


def test_ils_sparse_orthogonal():
    items = ItemList([1, 2, 3], ordered=True)
    row = [0, 1, 2]
    col = [0, 1, 2]
    data = [1, 1, 1]
    sparse = sps.csr_array((data, (row, col)), shape=(3, 3))

    result = intra_list_similarity(items, sparse)
    assert result == approx(0.0, abs=0.001)


def test_ils_partial_overlap():
    items = ItemList([1, 2, 3], ordered=True)
    # items 1 and 2 share category, item 3 different
    dense = np.array([[1, 0], [1, 0], [0, 1]])

    result = intra_list_similarity(items, dense)
    # similarity: (1,2)=1.0, (1,3)=0.0, (2,3)=0.0
    # average: (1.0 + 0.0 + 0.0) / 3 = 0.333
    assert result == approx(0.333, abs=0.01)


def test_ils_empty_list():
    items = ItemList([], ordered=True)
    dense = np.array([]).reshape(0, 3)

    result = intra_list_similarity(items, dense)
    assert np.isnan(result)


def test_ils_single_item():
    items = ItemList([1], ordered=True)
    dense = np.array([[1, 0, 0]])

    result = intra_list_similarity(items, dense)
    assert result == 1.0


def test_ils_two_items_similar():
    items = ItemList([1, 2], ordered=True)
    dense = np.array([[1, 0], [1, 0]])

    result = intra_list_similarity(items, dense)
    assert result == approx(1.0, abs=0.001)


def test_ils_two_items_different():
    items = ItemList([1, 2], ordered=True)
    dense = np.array([[1, 0], [0, 1]])

    result = intra_list_similarity(items, dense)
    assert result == approx(0.0, abs=0.001)


# ILS class


def test_ils_class_label(ml_ds):
    ils = ILS(ml_ds, "genres")
    assert ils.label == "ILS(genres)"

    ils10 = ILS(ml_ds, "genres", n=10)
    assert ils10.label == "ILS(genres)@10"


def test_ils_class_measure_list(ml_ds):
    ils = ILS(ml_ds, "genres", n=10)

    # get some items from ml_ds
    items = ml_ds.items.ids()[:15]
    recs = ItemList(items, ordered=True)
    truth = ItemList(items[:5])

    val = ils.measure_list(recs, truth)
    # ILS should be between 0 and 1
    assert 0.0 <= val <= 1.0


def test_ils_class_measure_list_empty(ml_ds):
    """Test ILS with empty recommendation list"""
    ils = ILS(ml_ds, "genres", n=10)

    recs = ItemList([], ordered=True)
    truth = ItemList([1, 2, 3])

    result = ils.measure_list(recs, truth)
    assert np.isnan(result)


def test_ils_large_sparse(ml_ds):
    items = ItemList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ordered=True)
    # 10 items, 5 categories, each item in unique category
    row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    col = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    categories = sps.csr_array((data, (row, col)), shape=(10, 5))

    ils = ILS(ml_ds, "genres", n=10)
    result = ils.measure_list(items, categories)
    assert 0.0 < result < 1.0


def test_ils_mixed_similarity(ml_ds):
    items = ItemList([1, 2, 3, 4], ordered=True)
    dense = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 0, 1], [0, 0.1, 0.9]])

    ils = ILS(ml_ds, "genres")
    result = ils.measure_list(items, dense)
    assert 0.0 < result < 1.0
