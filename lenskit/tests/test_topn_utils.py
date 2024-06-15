# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from lenskit.algorithms import CandidateSelector


def test_cs_rated_items_series():
    "rated_items should de-index series"
    items = ["a", "b", "wombat"]
    series = pd.Series(np.random.randn(3), index=items)

    i2 = CandidateSelector.rated_items(series)
    assert isinstance(i2, np.ndarray)
    assert all(i2 == items)


def test_cs_rated_items():
    "rated_items should return list as array"
    items = ["a", "b", "wombat"]

    i2 = CandidateSelector.rated_items(items)
    assert isinstance(i2, np.ndarray)
    assert all(i2 == items)


def test_cs_rated_items_array():
    "rated_items should return array as itself"
    items = ["a", "b", "wombat"]
    items = np.array(items)

    i2 = CandidateSelector.rated_items(items)
    assert isinstance(i2, np.ndarray)
    assert all(i2 == items)
    assert i2 is items
