# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
from scipy.sparse import coo_array, csr_array

from hypothesis import given
from pytest import mark, raises

from lenskit._accel import sparse_row_debug
from lenskit.data.matrix import (
    SparseIndexType,
    SparseRowArray,
    SparseRowType,
)
from lenskit.logging import get_logger
from lenskit.testing import sparse_arrays

_log = get_logger(__name__)


@given(sparse_arrays())
def test_sparse_from_csr(csr: csr_array[Any, tuple[int, int]]):
    arr = SparseRowArray.from_scipy(csr)
    assert isinstance(arr, SparseRowArray)
    assert isinstance(arr.type, SparseRowType)

    assert pa.types.is_list(arr.storage.type)
    assert len(arr) == csr.shape[0]
    assert arr.dimension == csr.shape[1]
    assert len(arr.indices) == csr.nnz
    assert len(arr.values) == csr.nnz
    assert np.all(arr.offsets.to_numpy() == csr.indptr)
    assert isinstance(arr.indices.type, SparseIndexType)


@given(sparse_arrays())
def test_sparse_row_indices(csr: csr_array[Any, tuple[int, int]]):
    arr = SparseRowArray.from_scipy(csr)
    assert isinstance(arr, SparseRowArray)
    assert isinstance(arr.type, SparseRowType)

    for i in range(csr.shape[0]):
        indices = arr.row_indices(i)
        row = csr[[i], :]
        assert len(indices) == row.nnz
        assert np.all(indices == row.indices)


@given(sparse_arrays())
def test_csr_to_rust(csr: csr_array[Any, tuple[int, int]]):
    "Test that we can send CSRs to Rust"
    arr = SparseRowArray.from_scipy(csr)
    assert isinstance(arr, SparseRowArray)
    assert isinstance(arr.type, SparseRowType)

    dt, nr, nc = sparse_row_debug(arr)
    _log.info("returned data type: %s", dt, nr=nr, nc=nc)
    assert (nr, nc) == csr.shape


@given(sparse_arrays())
def test_sparse_to_csr(csr: csr_array[Any, tuple[int, int]]):
    arr = SparseRowArray.from_scipy(csr)

    csr2 = arr.to_scipy()
    assert csr2.shape == csr.shape
    assert csr2.nnz == csr.nnz
    assert np.all(csr2.indptr == csr.indptr)
    assert np.all(csr2.indices == csr.indices)
    assert np.all(csr2.data == csr.data.astype("f4"))


@given(sparse_arrays())
def test_sparse_structure(csr: csr_array[Any, tuple[int, int]]):
    nr, nc = csr.shape

    arr = SparseRowArray.from_arrays(csr.indptr, csr.indices, shape=(nr, nc))
    assert len(arr) == nr
    assert arr.dimension == nc
    assert np.all(arr.offsets.to_numpy() == csr.indptr)
    assert arr.values is None


@given(sparse_arrays())
def test_sparse_structure_rust(csr: csr_array[Any, tuple[int, int]]):
    nr, nc = csr.shape

    arr = SparseRowArray.from_arrays(csr.indptr, csr.indices, shape=(nr, nc))

    dt, nr, nc = sparse_row_debug(arr)
    _log.info("returned data type: %s", dt, nr=nr, nc=nc)
    assert (nr, nc) == csr.shape


@given(sparse_arrays())
def test_sparse_from_legacy(csr: csr_array[Any, tuple[int, int]]):
    tbl = pa.ListArray.from_arrays(
        pa.array(csr.indptr.astype("i4")),
        pa.StructArray.from_arrays(
            [pa.array(csr.indices.astype("i4")), pa.array(csr.data)], ["index", "value"]
        ),
    )  # type: ignore
    nr, nc = csr.shape

    arr = SparseRowArray.from_array(tbl, nc)
    assert isinstance(arr, SparseRowArray)
    assert len(arr) == nr
    assert arr.dimension == nc


@given(sparse_arrays())
def test_sparse_from_legacy_i8(csr: csr_array[Any, tuple[int, int]]):
    tbl = pa.ListArray.from_arrays(
        pa.array(csr.indptr.astype("i4")),
        pa.StructArray.from_arrays(
            [pa.array(csr.indices.astype("i8")), pa.array(csr.data)], ["index", "value"]
        ),
    )  # type: ignore
    nr, nc = csr.shape

    arr = SparseRowArray.from_array(tbl, nc)
    assert isinstance(arr, SparseRowArray)
    assert len(arr) == nr
    assert arr.dimension == nc
    assert pa.types.is_int32(arr.storage.values.field(0).storage.type)


def test_sparse_type_direct_mismatch():
    t = SparseRowType(20)

    with raises(ValueError, match="mismatch"):
        SparseRowType.from_type(t, 30)


def test_sparse_type_extract_mismatch():
    t = SparseRowType(20)
    print("type", t)
    print("storage type", t.storage_type)

    with raises(ValueError, match="mismatch"):
        SparseRowType.from_type(t.storage_type, 30)


def test_sparse_type_extract_no_dimension():
    t = SparseRowType(20)
    print("type", t)
    print("storage type", t.storage_type)

    t2 = SparseRowType.from_type(t.storage_type)
    assert t2.dimension == t.dimension


def test_sparse_type_extract_missing_dimension():
    orig_t = pa.list_(
        pa.struct(
            [
                ("index", pa.int32()),
                ("value", pa.float32()),
            ]
        )
    )

    with raises(TypeError):
        SparseRowType.from_type(orig_t)


def test_sparse_type_bad_index():
    orig_t = pa.list_(
        pa.struct(
            [
                ("index", pa.int64()),
                ("value", pa.float32()),
            ]
        )
    )

    with raises(TypeError):
        SparseRowType.from_type(orig_t)


def test_sparse_type_not_struct():
    orig_t = pa.list_(pa.float32())

    with raises(TypeError):
        SparseRowType.from_type(orig_t)


def test_sparse_type_not_struct():
    orig_t = pa.list_(pa.float32())

    with raises(TypeError):
        SparseRowType.from_type(orig_t)


def test_sparse_type_not_list():
    orig_t = pa.struct(
        [
            ("index", pa.int64()),
            ("value", pa.float32()),
        ]
    )

    with raises(TypeError):
        SparseRowType.from_type(orig_t)
