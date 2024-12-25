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
from scipy.stats import kendalltau, permutation_test

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given, settings

from lenskit.basic import PopScorer, SoftmaxRanker
from lenskit.data.items import ItemList
from lenskit.testing import scored_lists

_log = logging.getLogger(__name__)


@given(scored_lists())
def test_unlimited_ranking(items: ItemList):
    topn = SoftmaxRanker()
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


@given(st.integers(min_value=1, max_value=100), scored_lists())
def test_configured_truncation(n, items: ItemList):
    topn = SoftmaxRanker(n)
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


@given(st.integers(min_value=1, max_value=100), scored_lists())
def test_runtime_truncation(n, items: ItemList):
    topn = SoftmaxRanker(rng="user")
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


def test_stochasticity(rng):
    "Test that softmax is varying but order-consistent"
    iids = np.arange(500)
    scores = rng.normal(size=500)
    scores = np.square(scores)
    items = ItemList(item_ids=iids, scores=scores)
    size = 50

    TRIALS = 100
    topn = SoftmaxRanker(n=size)

    _log.info("testing stochastic ranking: top %d of %d", size, len(items))

    ranks = np.full((size, TRIALS), -1, dtype=np.int64)
    scores = np.full((size, TRIALS), np.nan, dtype=np.float64)
    for i in range(TRIALS):
        ranked = topn(items)
        assert len(ranked) == size
        ranks[:, i] = ranked.ids()
        scores[:, i] = ranked.scores()

    id_counts = np.array([len(np.unique(ranks[i, :])) for i in range(size)])
    try:
        # at least half the positions should have more than 5 different items show up
        assert np.sum(id_counts < 5) <= size / 2
    except AssertionError as e:
        _log.info("failed test with n=%d, N=%d", size, len(items))
        _log.info("item counts: %s", id_counts)
        _log.info("items:\n%s", items.to_df())
        raise e

    # We want to test that it is usually putting things in the correct order.
    # We'll do this by computing Kendall's tau between the sampled ranking and
    # the scores of those items.  We want most of the lists (90%) to have
    # correlations statistically significnatly greater than zero (items tend to
    # be in the correct order).
    pvals = []
    taus = []
    for i in range(TRIALS):
        r_items = ranks[:, i]
        ranked = r_items >= 0
        r_items = r_items[ranked]

        rii = items[r_items]
        trs = size - np.arange(size)

        tau = kendalltau(trs, rii.scores(), alternative="greater")
        taus.append(tau.statistic)
        pvals.append(tau.pvalue)
        _log.info("trial %d: 𝜏=%.3f, p=%.3f", i, tau.statistic, tau.pvalue)

    pvals = np.array(pvals)
    _log.info("trial p-value statistics: mean=%.3f, median=%.3f", np.mean(pvals), np.median(pvals))
    # do 90% of trials pass the test?
    assert np.sum(pvals < 0.05) >= 0.9
