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

from pytest import approx

from lenskit.data import Dataset
from lenskit.testing import ml_ds, ml_ratings  # noqa: F401


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
