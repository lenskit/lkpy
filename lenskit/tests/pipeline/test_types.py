# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Tests for the pipeline type-checking functions.
"""

import typing
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import NoneType
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from pytest import warns

from lenskit.data import Dataset
from lenskit.data.matrix import MatrixDataset
from lenskit.pipeline.types import (
    TypecheckWarning,
    is_compatible_data,
    is_compatible_type,
    parse_type_string,
    type_string,
)

T = TypeVar("T")
Tstr = TypeVar("Tstr", bound=str)


def test_type_compat_identical():
    assert is_compatible_type(int, int)
    assert is_compatible_type(str, str)


def test_type_compat_subclass():
    assert is_compatible_type(MatrixDataset, Dataset)


def test_type_compat_assignable():
    assert is_compatible_type(int, float)


def test_type_raw_compat_with_generic():
    assert is_compatible_type(list, list[int])
    assert not is_compatible_type(set, list[int])


def test_type_compat_protocol():
    assert is_compatible_type(list, Sequence)
    assert is_compatible_type(list, typing.Sequence)
    assert not is_compatible_type(set, Sequence)
    assert not is_compatible_type(set, typing.Sequence)
    assert is_compatible_type(set, Iterable)


def test_type_compat_protocol_generic():
    assert is_compatible_type(list, Sequence[int])
    assert is_compatible_type(list, typing.Sequence[int])


def test_type_compat_generics_with_protocol():
    assert is_compatible_type(list[int], Sequence[int])


def test_type_incompat_generics():
    with warns(TypecheckWarning):
        assert is_compatible_type(list[int], list[str])
    with warns(TypecheckWarning):
        assert is_compatible_type(list[int], Sequence[str])


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


def test_numpy_typecheck():
    assert is_compatible_data(np.arange(10, dtype="i8"), NDArray[np.int64])
    assert is_compatible_data(np.arange(10, dtype="i4"), NDArray[np.int32])
    assert is_compatible_data(np.arange(10), ArrayLike)
    assert is_compatible_data(np.arange(10), NDArray[np.integer])
    # numpy types can be checked
    assert not is_compatible_data(np.arange(10), NDArray[np.float64])


def test_numpy_scalar_typecheck():
    assert is_compatible_data(np.int32(4270), np.integer[Any])


def test_numpy_scalar_typecheck2():
    assert is_compatible_data(np.int32(4270), np.integer[Any] | int)


def test_pandas_typecheck():
    assert is_compatible_data(pd.Series(["a", "b"]), ArrayLike)


def test_compat_with_typevar():
    assert is_compatible_data(100, T)


def test_not_compat_with_typevar():
    assert not is_compatible_data(100, Tstr)


def test_type_string_none():
    assert type_string(None) == "None"


def test_type_string_str():
    assert type_string(str) == "str"


def test_type_string_generic():
    assert type_string(list[str]) == "list"


def test_type_string_class():
    assert type_string(Path) == "pathlib.Path"


def test_parse_string_None():
    assert parse_type_string("None") == NoneType


def test_parse_string_int():
    assert parse_type_string("int") is int


def test_parse_string_class():
    assert parse_type_string("pathlib.Path") is Path


def test_parse_string_mod_class():
    assert parse_type_string("pathlib:Path") is Path
