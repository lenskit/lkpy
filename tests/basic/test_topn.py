# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
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

from lenskit.basic import PopScorer
from lenskit.basic.topn import TopNRanker
from lenskit.data.items import ItemList

_log = logging.getLogger(__name__)


@st.composite
def scored_lists(draw: st.DrawFn) -> ItemList:
    n = draw(st.integers(0, 1000))
    ids = np.arange(1, n + 1, dtype=np.int32)
    scores = draw(nph.arrays(nph.floating_dtypes(endianness="=", sizes=[32, 64]), n))
    return ItemList(item_ids=ids, scores=scores)


@given(scored_lists())
def test_unlimited_ranking(items: ItemList):
    topn = TopNRanker()
    ranked = topn(items=items)

    ids = items.ids()
    scores = items.scores("numpy")
    assert scores is not None
    invalid = np.isnan(scores)
    _log.info("ranking %d items, %d invalid", len(ids), np.sum(invalid))

    assert isinstance(ranked, ItemList)
    assert len(ranked) <= len(items)
    assert ranked.ordered
    # all valid items are included
    assert len(ranked) == np.sum(~invalid)

    # the set of valid items matches
    assert set(ids[~invalid]) == set(ranked.ids())

    # the scores match
    rank_s = ranked.scores("pandas", index="ids")
    assert rank_s is not None
    src_s = items.scores("pandas", index="ids")
    assert src_s is not None

    # make sure the scores were preserved properly
    rank_s, src_s = rank_s.align(src_s, "left")
    assert not np.any(np.isnan(src_s))
    assert np.all(rank_s == src_s)

    # make sure it's sorted
    assert np.all(rank_s.diff()) >= 0


@given(st.integers(min_value=1, max_value=100), scored_lists())
def test_configured_truncation(n, items: ItemList):
    topn = TopNRanker(n=n)
    ranked = topn(items=items)

    ids = items.ids()
    scores = items.scores("numpy")
    assert scores is not None
    invalid = np.isnan(scores)
    _log.info("top %d of %d items, %d invalid", n, len(ids), np.sum(invalid))

    val_items = items[~invalid]

    assert isinstance(ranked, ItemList)
    assert ranked.ordered
    assert len(ranked) == min(n, len(val_items))

    # the scores match
    rank_s = ranked.scores("pandas", index="ids")
    assert rank_s is not None
    src_s = items.scores("pandas", index="ids")
    assert src_s is not None
    src_s = src_s[src_s.notna()]

    # make sure the scores were preserved properly
    rank_s, src_s = rank_s.align(src_s, "left")
    assert not np.any(np.isnan(src_s))
    assert np.all(rank_s == src_s)

    # make sure it's sorted
    assert np.all(rank_s.diff()) >= 0

    # make sure it's the largest
    omitted = ~np.isin(items.ids(), ranked.ids())
    if np.any(omitted) and np.any(~np.isnan(scores[omitted])):
        assert np.all(rank_s >= np.nanmax(scores[omitted]))


@given(st.integers(min_value=1, max_value=100), scored_lists())
def test_runtime_truncation(n, items: ItemList):
    topn = TopNRanker()
    ranked = topn(items=items, n=n)

    ids = items.ids()
    scores = items.scores("numpy")
    assert scores is not None
    invalid = np.isnan(scores)
    _log.info("top %d of %d items, %d invalid", n, len(ids), np.sum(invalid))

    val_items = items[~invalid]

    assert isinstance(ranked, ItemList)
    assert ranked.ordered
    assert len(ranked) == min(n, len(val_items))

    # the scores match
    rank_s = ranked.scores("pandas", index="ids")
    assert rank_s is not None
    src_s = items.scores("pandas", index="ids")
    assert src_s is not None
    src_s = src_s[src_s.notna()]

    # make sure the scores were preserved properly
    rank_s, src_s = rank_s.align(src_s, "left")
    assert not np.any(np.isnan(src_s))
    assert np.all(rank_s == src_s)

    # make sure it's sorted
    assert np.all(rank_s.diff()) >= 0

    # make sure it's the largest
    omitted = ~np.isin(items.ids(), ranked.ids())
    if np.any(omitted) and np.any(~np.isnan(scores[omitted])):
        assert np.all(rank_s >= np.nanmax(scores[omitted]))
