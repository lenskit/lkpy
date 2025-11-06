# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Tests for data set information
"""

import re

import numpy as np
import pandas as pd
import pyarrow as pa

from pytest import approx

from lenskit.data import Dataset
from lenskit.data.attributes import (
    ListAttributeSet,
    ScalarAttributeSet,
    SparseAttributeSet,
    VectorAttributeSet,
)
from lenskit.testing import ml_ds, ml_ratings  # noqa: F401

from ..movielens.test_ml20m import ml20m


def test_attribute_qname(ml_ds: Dataset):
    title_attr = ml_ds.entities("item").attribute("title")
    assert title_attr.name == "title"
    assert title_attr._qname == "item.title"


def test_attribute_str(ml_ds: Dataset):
    title_attr = ml_ds.entities("item").attribute("title")
    assert re.match(r"^<ScalarAttributeSet item\.title: string \(\d+ entities\)>", str(title_attr))


def test_attribute_repr(ml_ds: Dataset):
    title_attr = ml_ds.entities("item").attribute("title")
    assert re.match(r"^<ScalarAttributeSet item\.title {", repr(title_attr))


def test_scalar_type(ml_ds: Dataset):
    title_attr = ml_ds.entities("item").attribute("title")
    assert title_attr.layout.value == "scalar"
    assert isinstance(title_attr, ScalarAttributeSet)
    assert title_attr.data_type == pa.string()


def test_list_type(ml_ds: Dataset):
    attr = ml_ds.entities("item").attribute("genres")
    assert attr.layout.value == "list"
    assert isinstance(attr, ListAttributeSet)
    assert attr.data_type == pa.string()


def test_vector_dtype(ml20m: Dataset):
    attr = ml20m.entities("item").attribute("tag_genome")
    assert attr.layout.value == "vector"
    assert isinstance(attr, VectorAttributeSet)
    assert pa.types.is_floating(attr.data_type)


def test_sparse_dtype(ml_ds: Dataset):
    attr = ml_ds.entities("item").attribute("tag_counts")
    assert attr.layout.value == "sparse"
    assert isinstance(attr, SparseAttributeSet)
    assert pa.types.is_integer(attr.data_type)
