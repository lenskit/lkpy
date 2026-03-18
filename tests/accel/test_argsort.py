# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pyarrow as pa

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given
from pytest import mark

from lenskit._accel import data


@given(
    nph.arrays(
        nph.floating_dtypes(endianness="="),
        nph.array_shapes(max_dims=1),
        elements={"allow_nan": False, "allow_infinity": False},
    )
)
def test_sort_floats(arr):
    a2 = pa.array(arr)

    idx = data.argsort_descending(a2)
    assert len(idx) == len(arr)
    assert set(idx.to_pylist()) == set(range(len(arr)))

    items = arr[idx]

    # descending order
    for i in range(1, len(arr)):
        assert items[i] <= items[i - 1]


@given(
    nph.arrays(
        nph.integer_dtypes(endianness="="),
        nph.array_shapes(max_dims=1),
        elements={"allow_nan": False, "allow_infinity": False},
    )
)
def test_sort_ints(arr):
    a2 = pa.array(arr)

    idx = data.argsort_descending(a2)
    assert len(idx) == len(arr)
    assert set(idx.to_pylist()) == set(range(len(arr)))

    items = arr[idx]

    # descending order
    for i in range(1, len(arr)):
        assert items[i] <= items[i - 1]


@given(
    nph.arrays(
        nph.floating_dtypes(endianness="="),
        nph.array_shapes(max_dims=1),
        elements={"allow_nan": False, "allow_infinity": False},
    ),
    st.integers(min_value=1, max_value=500),
)
def test_topn_floats(arr, n):
    a2 = pa.array(arr)
    tgt_n = min(len(arr), n)

    idx = data.argtopn(a2, n).to_numpy()
    assert len(idx) == tgt_n
    assert np.all(idx >= 0)
    assert np.all(idx < len(arr))

    items = arr[idx]

    # descending order
    for i in range(1, tgt_n):
        assert items[i] <= items[i - 1]

    mask = np.ones(len(arr), dtype=np.bool_)
    mask[idx] = False
    nopes = arr[mask]
    assert np.all(nopes <= np.min(items))


@mark.xfail(reason="NaN not yet supported")
@given(
    nph.arrays(
        nph.floating_dtypes(endianness="="),
        nph.array_shapes(max_dims=1),
        elements={"allow_nan": True, "allow_infinity": False},
    ),
    st.integers(min_value=1, max_value=500),
)
def test_topn_with_nans(arr, n):
    a2 = pa.array(arr)
    tgt_n = min(n, len(arr) - a2.null_count)

    idx = data.argtopn(a2, n).to_numpy()
    assert len(idx) == tgt_n
    assert np.all(idx >= 0)
    assert np.all(idx < len(arr))

    items = arr[idx]
    assert not np.any(np.isnan(items))

    # descending order
    for i in range(1, tgt_n):
        assert items[i] <= items[i - 1]

    mask = np.ones(len(arr), dtype=np.bool_)
    mask[idx] = False
    mask[np.isnan(arr)] = False
    nopes = arr[mask]
    assert np.all(nopes <= np.min(items))


@given(
    nph.arrays(
        nph.floating_dtypes(endianness="="),
        nph.array_shapes(max_dims=1),
        elements={"allow_nan": True, "allow_infinity": False},
    ),
    st.integers(min_value=1, max_value=500),
)
def test_topn_with_nulls(arr, n):
    a2 = pa.array(arr, mask=np.isnan(arr))
    tgt_n = min(n, len(arr) - a2.null_count)

    idx = data.argtopn(a2, n).to_numpy()
    assert len(idx) == tgt_n
    assert np.all(idx >= 0)
    assert np.all(idx < len(arr))

    items = arr[idx]
    assert not np.any(np.isnan(items))

    # descending order
    for i in range(1, tgt_n):
        assert items[i] <= items[i - 1]

    if len(items):
        mask = np.ones(len(arr), dtype=np.bool_)
        mask[idx] = False
        mask[np.isnan(arr)] = False
        nopes = arr[mask]
        assert np.all(nopes <= np.min(items))


@given(
    nph.arrays(
        nph.floating_dtypes(endianness="="),
        nph.array_shapes(max_dims=1),
        elements={"allow_nan": False, "allow_infinity": True},
    ),
    st.integers(min_value=1, max_value=500),
)
def test_topn_with_inf(arr, n):
    a2 = pa.array(arr)
    tgt_n = min(n, len(arr) - a2.null_count)

    idx = data.argtopn(a2, n).to_numpy()
    assert len(idx) == tgt_n
    assert np.all(idx >= 0)
    assert np.all(idx < len(arr))

    items = arr[idx]
    assert not np.any(np.isnan(items))

    # descending order
    for i in range(1, tgt_n):
        assert items[i] <= items[i - 1]

    mask = np.ones(len(arr), dtype=np.bool_)
    mask[idx] = False
    mask[np.isnan(arr)] = False
    nopes = arr[mask]
    assert np.all(nopes <= np.min(items))
