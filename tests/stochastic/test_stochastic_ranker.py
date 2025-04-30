# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.distance import jaccard
from scipy.stats import kendalltau, permutation_test

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given, settings
from pytest import mark

from lenskit import batch
from lenskit.als import ImplicitMFScorer
from lenskit.basic import PopScorer
from lenskit.basic.topn import TopNRanker
from lenskit.data import Dataset, ItemList
from lenskit.logging import get_logger
from lenskit.pipeline import topn_pipeline
from lenskit.splitting import simple_test_pair
from lenskit.stochastic import StochasticTopNRanker
from lenskit.testing import BasicComponentTests, ScorerTests, scored_lists

_log = get_logger(__name__)


class TestSoftmax(BasicComponentTests):
    component = StochasticTopNRanker


@mark.filterwarnings("error:divide by zero")
@given(scored_lists(), st.sampled_from(["linear", "softmax"]))
def test_unlimited_ranking(items: ItemList, transform):
    topn = StochasticTopNRanker(transform=transform)
    ranked = topn(items=items)

    ids = items.ids()
    scores = items.scores("numpy")
    assert scores is not None
    invalid = ~np.isfinite(scores)

    try:
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
    except AssertionError as e:
        e.add_note("ranked {} items ({} invalid)".format(len(ids), np.sum(invalid)))
        raise e


@given(st.integers(min_value=1, max_value=100), scored_lists())
def test_configured_truncation(n, items: ItemList):
    topn = StochasticTopNRanker(n=n)
    ranked = topn(items=items)

    ids = items.ids()
    scores = items.scores("numpy")
    assert scores is not None
    invalid = ~np.isfinite(scores)
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
    topn = StochasticTopNRanker(rng="user")
    ranked = topn(items=items, n=n)

    ids = items.ids()
    scores = items.scores("numpy")
    assert scores is not None
    invalid = ~np.isfinite(scores)
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


@given(scored_lists(n=st.integers(100, 5000), scores=st.floats(0, 15)), st.floats(0.1, 10000))
def test_overflow(items: ItemList, scale: float):
    topn = StochasticTopNRanker(transform="softmax", scale=scale)
    ranked = topn(items=items, include_weights=True)

    ids = items.ids()
    scores = items.scores("numpy")
    assert scores is not None
    assert np.all(np.isfinite(scores))

    k2 = topn._compute_keys(scores, topn._rng_factory(None))
    assert np.all(np.isfinite(k2))

    try:
        assert isinstance(ranked, ItemList)
        assert len(ranked) == len(items)
        assert ranked.ordered

        weights = ranked.field("weight")
        assert weights is not None
        assert np.all(np.isfinite(weights))

        # the scores match
        rank_s = ranked.scores("pandas", index="ids")
        assert rank_s is not None
        src_s = items.scores("pandas", index="ids")
        assert src_s is not None

        # make sure the scores were preserved properly
        rank_s, src_s = rank_s.align(src_s, "left")
        assert not np.any(np.isnan(src_s))
        assert np.all(rank_s == src_s)
    except AssertionError as e:
        e.add_note("ranked {} items".format(len(ids)))
        raise e


def test_stochasticity(rng):
    "Test that softmax is varying but order-consistent"
    iids = np.arange(500)
    scores = rng.normal(size=500)
    scores = np.square(scores)
    items = ItemList(item_ids=iids, scores=scores)
    size = 50

    TRIALS = 100
    topn = StochasticTopNRanker(n=size)

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
    assert np.mean(pvals < 0.05) >= 0.9


def test_scale_affects_ranking(rng, ml_ds: Dataset):
    split = simple_test_pair(ml_ds, f_rates=0.5, rng=rng)
    pipe = topn_pipeline(ImplicitMFScorer())
    pipe.train(split.train)
    recs = batch.recommend(pipe, split.test, n_jobs=1)

    topn = TopNRanker()
    samp_frac = StochasticTopNRanker(scale=0.01)
    samp_one = StochasticTopNRanker(scale=1)
    samp_hundred = StochasticTopNRanker(scale=100)

    jc_frac = []
    jc_one = []
    jc_hundred = []

    for uid, ilist in recs:
        rl_topn = topn(items=ilist, n=10)
        rl_frac = samp_frac(items=ilist, n=10)
        rl_one = samp_one(items=ilist, n=10)
        rl_hundred = samp_hundred(items=ilist, n=10)

        mask_topn = np.zeros(ml_ds.item_count, dtype=np.bool_)
        mask_topn[rl_topn.numbers(vocabulary=ml_ds.items)] = True
        mask_frac = np.zeros(ml_ds.item_count, dtype=np.bool_)
        mask_frac[rl_frac.numbers(vocabulary=ml_ds.items)] = True
        mask_one = np.zeros(ml_ds.item_count, dtype=np.bool_)
        mask_one[rl_one.numbers(vocabulary=ml_ds.items)] = True
        mask_hundred = np.zeros(ml_ds.item_count, dtype=np.bool_)
        mask_hundred[rl_hundred.numbers(vocabulary=ml_ds.items)] = True

        jc_frac.append(jaccard(mask_topn, mask_frac))
        jc_one.append(jaccard(mask_topn, mask_one))
        jc_hundred.append(jaccard(mask_topn, mask_hundred))

    # high-temp should agree less than flat
    assert np.mean(jc_frac) < np.mean(jc_one)
    # low-temp should agree more than flat
    assert np.mean(jc_hundred) < np.mean(jc_one)
