# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
from scipy.sparse import csr_array

import pytest

from lenskit.data.matrix import normalize_matrix


def test_normalize_unit():
    # sparse matrix
    sparse = csr_array(
        (
            np.array([3.0, 4.0, 1.0, 1.0], dtype=np.float32),
            np.array([0, 1, 0, 1]),
            np.array([0, 2, 4]),
        ),
        shape=(2, 2),
    )
    result = normalize_matrix(sparse, normalize="unit")
    row_norms = np.sqrt(np.array(result.multiply(result).sum(axis=1)))
    assert np.allclose(row_norms, 1.0)
    assert np.allclose(result[0].toarray(), [[0.6, 0.8]])

    # dense matrix
    dense = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    result = normalize_matrix(dense, normalize="unit")
    row_norms = np.linalg.norm(result, axis=1)
    assert np.allclose(row_norms, 1.0)
    assert np.allclose(result[0], [0.6, 0.8])

    # zero rows should remain zero
    sparse_zero = csr_array(
        (
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([0, 1]),
            np.array([0, 2, 2]),
        ),
        shape=(2, 2),
    )
    result = normalize_matrix(sparse_zero, normalize="unit")
    assert np.allclose(result[1].toarray(), [[0.0, 0.0]])


def test_normalize_distribution():
    sparse = csr_array(
        (
            np.array([1.0, 2.0, 3.0, 1.0, 1.0], dtype=np.float32),
            np.array([0, 1, 2, 0, 1]),
            np.array([0, 3, 5]),
        ),
        shape=(2, 3),
    )
    result = normalize_matrix(sparse, normalize="distribution")
    row_sums = np.array(result.sum(axis=1))
    assert np.allclose(row_sums, 1.0)
    assert np.allclose(result[0].toarray(), [[1 / 6, 2 / 6, 3 / 6]])

    dense = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    result = normalize_matrix(dense, normalize="distribution")
    row_sums = result.sum(axis=1)
    assert np.allclose(row_sums, 1.0)
    assert np.allclose(result[0], [1 / 6, 2 / 6, 3 / 6])

    # negative values should raise error
    sparse_neg = csr_array(
        (
            np.array([1.0, -2.0], dtype=np.float32),
            np.array([0, 1], dtype=np.int32),
            np.array([0, 2], dtype=np.int32),
        ),
        shape=(1, 2),
    )
    with pytest.raises(ValueError, match="Cannot normalize"):
        normalize_matrix(sparse_neg, normalize="distribution")

    dense_neg = np.array([[1.0, -2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="Cannot normalize"):
        normalize_matrix(dense_neg, normalize="distribution")


def test_normalize_none():
    sparse = csr_array(
        (
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([0, 1], dtype=np.int32),
            np.array([0, 2], dtype=np.int32),
        ),
        shape=(1, 2),
    )
    result = normalize_matrix(sparse, normalize=None)
    assert result is sparse

    dense = np.array([[1.0, 2.0]], dtype=np.float32)
    result = normalize_matrix(dense, normalize=None)
    assert result is dense
