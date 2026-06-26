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


def test_importable_string_none():
    assert make_importable_path(None) == "None"


def test_importable_string_str():
    assert make_importable_path(str) == "str"


def test_importable_string_generic():
    assert make_importable_path(list[str]) == "list"


def test_importable_string_class():
    assert make_importable_path(UUID) == "uuid:UUID"


def test_importable_string_function():
    from lenskit.config import configure

    path = make_importable_path(configure)
    assert path == "lenskit.config:configure"

    func = import_path_string(path)
    assert func is configure


def test_importable_string_private_module():
    from lenskit.pipeline import Pipeline

    path = make_importable_path(Pipeline)
    assert path == "lenskit.pipeline:Pipeline"

    cls = import_path_string(path)
    assert cls is Pipeline


def test_importable_string_private_function():
    from lenskit.logging import stdout_console

    path = make_importable_path(stdout_console)
    assert path == "lenskit.logging:stdout_console"

    func = import_path_string(path)
    assert func is stdout_console


def test_importable_component_private_function():
    from lenskit.als import BiasedMFScorer

    path = make_importable_path(BiasedMFScorer)
    assert path == "lenskit.als:BiasedMFScorer"

    func = import_path_string(path)
    assert func is BiasedMFScorer


def test_parse_string_None():
    assert import_path_string("None") == NoneType


def test_parse_string_int():
    assert import_path_string("int") is int


def test_parse_string_class():
    assert import_path_string("pathlib.Path") is Path


def test_parse_string_mod_class():
    assert import_path_string("pathlib:Path") is Path
