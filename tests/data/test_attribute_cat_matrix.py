# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import json

import numpy as np
import pyarrow as pa
from scipy.sparse import csr_array

from lenskit.data.attributes import (
    ListAttributeSet,
    ScalarAttributeSet,
    SparseAttributeSet,
    VectorAttributeSet,
)
from lenskit.data.matrix import SparseRowArray
from lenskit.data.schema import AttrLayout, ColumnSpec
from lenskit.data.vocab import Vocabulary


def test_scalar_cat_matrix():
    data = [1, 2, 1, 3, 2]
    arr = pa.array(data, type=pa.int32())
    table = pa.Table.from_arrays([arr], names=["attr"])

    spec = ColumnSpec(layout=AttrLayout.SCALAR)
    vocab = Vocabulary(pa.array([1, 2, 3]))
    attr_set = ScalarAttributeSet("attr", spec, table, vocab, None)

    matrix, cat_vocab = attr_set.cat_matrix(normalize="unit")
    assert isinstance(matrix, csr_array)
    assert matrix.shape == (5, len(cat_vocab))
    assert len(cat_vocab) == 3


def test_list_cat_matrix():
    data = [[1, 2], [2, 3], [1], [3], [1, 2]]
    arr = pa.array(data, type=pa.list_(pa.int32()))
    table = pa.Table.from_arrays([arr], names=["attr"])

    spec = ColumnSpec(layout=AttrLayout.LIST)
    vocab = Vocabulary(pa.array([1, 2, 3], type=pa.int32()))
    attr_set = ListAttributeSet("attr", spec, table, vocab, None)

    matrix, cat_vocab = attr_set.cat_matrix(normalize="distribution")
    assert isinstance(matrix, csr_array)
    assert matrix.shape == (5, len(cat_vocab))
    assert len(cat_vocab) == 3


def test_vector_cat_matrix():
    data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [0.5, 0.5, 0.5],
        [1.0, 0.0, 1.0],
    ]
    arr = pa.array(data, type=pa.list_(pa.float32()))
    table = pa.Table.from_arrays([arr], names=["attr"])

    spec = ColumnSpec(layout=AttrLayout.VECTOR, vector_size=3)
    vocab = Vocabulary(pa.array([0, 1, 2, 3, 4]))
    attr_set = VectorAttributeSet("attr", spec, table, vocab, None)

    matrix, cat_vocab = attr_set.cat_matrix()
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (5, 3)
    assert cat_vocab is None


def test_sparse_cat_matrix():
    mat = csr_array(
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 3, 2, 0, 3]),
            np.array([0, 2, 3, 5]),
        ),
        shape=(3, 4),
    )

    sra = SparseRowArray.from_scipy(mat)

    table = pa.Table.from_arrays([sra], names=["attr"])
    spec = ColumnSpec(layout=AttrLayout.SPARSE)
    vocab = Vocabulary(pa.array([1, 2, 3]))
    attr_set = SparseAttributeSet("attr", spec, table, vocab, None)

    matrix, cat_vocab = attr_set.cat_matrix(normalize="unit")

    assert isinstance(matrix, csr_array)
    assert matrix.shape == (3, 4)
    assert cat_vocab is None


def test_sparse_vocab_cat_matrix():
    mat = csr_array(
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 3, 2, 0, 3]),
            np.array([0, 2, 3, 5]),
        ),
        shape=(3, 4),
    )

    sra = SparseRowArray.from_scipy(mat)

    field = pa.field(
        "attr",
        sra.type,
        metadata={b"lenskit:names": json.dumps(["d0", "d1", "d2", "d3"]).encode()},
    )
    table = pa.Table.from_arrays([sra], schema=pa.schema([field]))

    spec = ColumnSpec(layout=AttrLayout.SPARSE)
    entity_vocab = Vocabulary(pa.array([1, 2, 3]))  # entity vocabulary
    attr_set = SparseAttributeSet("attr", spec, table, entity_vocab, None)

    matrix, cat_vocab = attr_set.cat_matrix(normalize="unit")

    assert isinstance(matrix, csr_array)
    assert matrix.shape == (3, 4)
    assert isinstance(cat_vocab, Vocabulary)
    assert len(cat_vocab) == 4
