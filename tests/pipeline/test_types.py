# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Tests for the pipeline type-checking functions.
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


def test_is_not_instance():
    assert not is_instance_or_subclass(10, str)


def test_is_not_subclass():
    assert not is_instance_or_subclass(int, str)


def test_is_instance():
    assert is_instance_or_subclass("foo", str)


def test_is_instance_proto():
    assert is_instance_or_subclass([], Sequence)
    assert not is_instance_or_subclass(10, Sequence)


def test_is_subclass_proto():
    assert is_instance_or_subclass(list, Sequence)
