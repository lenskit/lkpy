# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given, settings

import lenskit.util.test as lktu
from lenskit.basic import PopScorer
from lenskit.basic.random import RandomSelector
from lenskit.basic.topn import TopNRanker
from lenskit.data.items import ItemList
from lenskit.util.test import ml_ds, ml_ratings  # noqa: F401

_log = logging.getLogger(__name__)


@st.composite
def scored_lists(draw: st.DrawFn) -> ItemList:
    n = draw(st.integers(0, 1000))
    ids = np.arange(1, n + 1, dtype=np.int32)
    scores = draw(nph.arrays(nph.floating_dtypes(endianness="=", sizes=[32, 64]), n))
    return ItemList(item_ids=ids, scores=scores)


@given(scored_lists())
def test_unlimited_selection(items: ItemList):
    rsel = RandomSelector()
    ranked = rsel(items=items)

    ids = items.ids()

    assert len(ranked) == len(ids)
    assert set(ranked.ids()) == set(ids)


@given(st.integers(min_value=1, max_value=100), scored_lists())
def test_configured_truncation(n, items: ItemList):
    rsel = RandomSelector(n)
    ranked = rsel(items=items)

    assert len(ranked) == min(n, len(items))
    # all items are unique
    assert len(set(ranked.ids())) == len(ranked)


@given(st.integers(min_value=1, max_value=100), scored_lists())
def test_runtime_truncation(n, items: ItemList):
    rsel = RandomSelector()
    ranked = rsel(items=items, n=n)

    assert len(ranked) == min(n, len(items))
    # all items are unique
    assert len(set(ranked.ids())) == len(ranked)
