# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Tests for is_compatible_type
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


def test_type_compat_identical():
    assert is_compatible_type(int, int)
    assert is_compatible_type(str, str)


def test_type_compat_subclass():
    assert is_compatible_type(MatrixRelationshipSet, RelationshipSet)


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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"cannot type-check generic", TypecheckWarning)
        assert is_compatible_type(list[int], Sequence[int])


def test_type_incompat_generics():
    with warns(TypecheckWarning):
        assert is_compatible_type(list[int], list[str])
    with warns(TypecheckWarning):
        assert is_compatible_type(list[int], Sequence[str])


def test_compatible_type_any():
    assert is_compatible_type(int, Any)


def test_query_subtype():
    assert is_compatible_type(RecQuery, QueryInput)
