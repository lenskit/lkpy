# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Tests for is_compatible_data
"""

import typing
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import NoneType
from typing import Any, TypeVar, Union
from uuid import UUID

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from pytest import mark, warns

from lenskit.data import MatrixRelationshipSet, QueryInput, RecQuery, RelationshipSet
from lenskit.pipeline._types import (
    TypecheckWarning,
    import_path_string,
    is_compatible_data,
    is_compatible_type,
    is_instance_or_subclass,
    make_importable_path,
)

T = TypeVar("T")
Tstr = TypeVar("Tstr", bound=str)


def test_data_compat_basic():
    assert is_compatible_data(72, int)
    assert is_compatible_data("hello", str)
    assert not is_compatible_data(72, str)


def test_data_compat_float_assignabile():
    assert is_compatible_data(72, float)


def test_data_compat_generic():
    assert is_compatible_data(["foo"], list[str])
    # this is compatible because we can't check generics
    # with warns(TypecheckWarning):
    assert is_compatible_data([72], list[str])


def test_numpy_typecheck_NDArray():
    assert is_compatible_data(np.arange(10), NDArray[np.integer])
    assert is_compatible_data(np.arange(10, dtype="i8"), NDArray[np.int64])
    assert is_compatible_data(np.arange(10, dtype="i4"), NDArray[np.int32])
    # numpy types can be checked
    assert not is_compatible_data(np.arange(10), NDArray[np.float64])


def test_numpy_typecheck_ndarray():
    assert is_compatible_data(np.arange(10), np.ndarray[Any, np.dtype[np.integer]])
    assert is_compatible_data(np.arange(10, dtype="i8"), np.ndarray[Any, np.dtype[np.int64]])
    assert is_compatible_data(np.arange(10, dtype="i4"), np.ndarray[Any, np.dtype[np.int32]])
    # numpy types can be checked
    assert not is_compatible_data(np.arange(10), np.ndarray[Any, np.dtype[np.float64]])


def test_numpy_array_like():
    assert is_compatible_data(np.arange(10), ArrayLike)


def test_numpy_scalar_typecheck():
    assert is_compatible_data(np.int32(4270), np.integer[Any])


def test_numpy_scalar_typecheck2():
    assert is_compatible_data(np.int32(4270), np.integer[Any] | int)


def test_compatible_data_union():
    assert is_compatible_data("foo", str | bytes)
    assert is_compatible_data("foo", Union[str, bytes])


def test_compatible_any():
    assert is_compatible_data(50, Any)


@mark.skip("broke with NumPy 2.4")
def test_pandas_typecheck():
    assert is_compatible_data(pd.Series(["a", "b"]), ArrayLike)


def test_compat_with_typevar():
    assert is_compatible_data(100, T)


def test_not_compat_with_typevar():
    assert not is_compatible_data(100, Tstr)


def test_query_valid():
    query = RecQuery(user_id=47)
    assert is_compatible_data(query, RecQuery)
    assert is_compatible_data(query, QueryInput)
