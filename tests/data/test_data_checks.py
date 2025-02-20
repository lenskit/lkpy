# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given
from pytest import raises

from lenskit.data.checks import check_1d, check_type


@given(nph.arrays(nph.scalar_dtypes(), nph.array_shapes(min_dims=1, max_dims=1)))
def test_check_1d_ok(arr):
    check_1d(arr)


@given(nph.arrays(nph.scalar_dtypes(), nph.array_shapes(min_dims=1, max_dims=1)))
def test_check_1d_ok_return(arr):
    assert check_1d(arr, error="return")


@given(nph.arrays(nph.scalar_dtypes(), nph.array_shapes(min_dims=2)))
def test_check_1d_bad(arr):
    with raises(TypeError, match="must be 1D"):
        check_1d(arr)


@given(nph.arrays(nph.scalar_dtypes(), nph.array_shapes(min_dims=2)))
def test_check_1d_bad_return(arr):
    assert not check_1d(arr, error="return")


@given(nph.arrays(nph.scalar_dtypes(), nph.array_shapes()), st.integers(min_value=0))
def test_check_expected_size(arr, exp):
    if arr.shape == (exp,):
        check_1d(arr, exp)
    else:
        with raises(TypeError):
            check_1d(arr, exp)


@given(nph.arrays(nph.scalar_dtypes(), nph.array_shapes()), st.integers(min_value=0))
def test_check_expected_size_return(arr, exp):
    if arr.shape == (exp,):
        assert check_1d(arr, exp, error="return")
    else:
        assert not check_1d(arr, exp, error="return")


@given(nph.arrays(nph.scalar_dtypes(), nph.array_shapes()))
def test_check_type_ok(arr):
    check_type(arr, arr.dtype.type)


@given(nph.arrays(nph.floating_dtypes(), nph.array_shapes()))
def test_check_type_ok_subclass(arr):
    check_type(arr, np.floating)


@given(nph.arrays(st.one_of(nph.integer_dtypes(), nph.floating_dtypes()), nph.array_shapes()))
def test_check_type_ok_multi(arr):
    check_type(arr, np.integer, np.floating)


@given(nph.arrays(nph.scalar_dtypes(), nph.array_shapes()))
def test_check_type_ok_return(arr):
    assert check_type(arr, arr.dtype.type, error="return")


@given(nph.arrays(nph.floating_dtypes(), nph.array_shapes()))
def test_check_type_bad_float(arr):
    with raises(TypeError):
        check_type(arr, np.integer)


@given(nph.arrays(nph.floating_dtypes(), nph.array_shapes()))
def test_check_type_bad_float_return(arr):
    assert not check_type(arr, np.integer, error="return")


@given(nph.arrays(nph.integer_dtypes(), nph.array_shapes()))
def test_check_type_bad_int(arr):
    with raises(TypeError):
        check_type(arr, np.floating)


@given(nph.arrays(nph.unicode_string_dtypes(), nph.array_shapes()))
def test_check_type_bad_str(arr):
    with raises(TypeError):
        check_type(arr, np.number)

    with raises(TypeError):
        check_type(arr, np.integer, np.floating)
