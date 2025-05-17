# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given

from lenskit.data import ItemList
from lenskit.testing import scored_lists


def test_topn_empty():
    il = ItemList(scores=0)

    top = il.top_n()
    assert top.ordered
    assert len(top) == 0


@given(scored_lists())
def test_top_all(items):
    top = items.top_n()
    assert len(top) == len(items) - np.sum(np.isnan(items.scores()))
    assert top.ordered

    scores = top.scores()
    assert scores is not None
    diffs = np.diff(scores)
    assert np.all((diffs <= 0) | np.isnan(diffs))


@given(scored_lists(), st.integers(min_value=1))
def test_top_n(items, n):
    top = items.top_n(n)
    assert len(top) == min(n, len(items) - np.sum(np.isnan(items.scores())))
    assert top.ordered

    scores = top.scores()
    assert scores is not None
    diffs = np.diff(scores)
    assert np.all((diffs <= 0) | np.isnan(diffs))


@given(scored_lists(), st.integers(min_value=1))
def test_top_n_field(items, n):
    items = ItemList(items, scores=False, rating=items.scores())
    top = items.top_n(n, scores="rating")
    assert len(top) == min(n, len(items) - np.sum(np.isnan(items.field("rating"))))
    assert top.ordered

    rates = top.field("rating")
    assert rates is not None
    diffs = np.diff(rates)
    assert np.all((diffs <= 0) | np.isnan(diffs))


@given(scored_lists(), st.integers(min_value=1))
def test_top_n_keys(items, n):
    keys = np.random.randn(len(items))
    top = items.top_n(n, scores=keys)
    assert len(top) == min(n, len(items))
    assert top.ordered

    # the items should correspond to keys in decreasing order
    keys = pd.Series(keys, index=items.ids())
    keys = keys.reindex(top.ids())
    diffs = np.diff(keys)
    assert np.all((diffs <= 0) | np.isnan(diffs))
