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
from scipy.special import softmax
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
from lenskit.operations import recommend
from lenskit.pipeline import topn_pipeline
from lenskit.splitting import simple_test_pair
from lenskit.stochastic import StochasticTopNRanker
from lenskit.testing import BasicComponentTests, ScorerTests, scored_lists
from lenskit.training import TrainingOptions

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


@given(
    scored_lists(
        n=st.integers(100, 5000), scores=st.floats(-5, 15, width=32, allow_infinity=False)
    ),
    st.floats(0.1, 10000),
)
def test_overflow(items: ItemList, scale: float):
    topn = StochasticTopNRanker(transform="softmax", scale=scale)
    ranked = topn(items=items, include_weights=True)

    ids = items.ids()
    scores = items.scores("numpy")
    assert scores is not None
    assert np.all(np.isfinite(scores))

    sm = softmax(scores)
    assert np.all(np.isfinite(sm))
    assert np.all(sm >= 0)
    assert np.all(sm <= 1)

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


@mark.flaky(retries=3)
def test_scale_affects_ranking(ml_ds: Dataset):
    """
    Test that different softmax scales produce different levels of ranking variation.
    """
    rng = np.random.default_rng()
    pipe = topn_pipeline(ImplicitMFScorer(embedding_size=32, weight=5))
    pipe.train(ml_ds, TrainingOptions(rng=rng))

    seed1, seed2, seed3 = rng.bit_generator.random_raw(3)

    topn = TopNRanker()
    samp_frac = StochasticTopNRanker(scale=0.01, rng=int(seed1))
    samp_one = StochasticTopNRanker(scale=1, rng=int(seed2))
    samp_hundred = StochasticTopNRanker(scale=50, rng=int(seed3))

    jc_frac = []
    jc_one = []
    jc_hundred = []

    for uid in rng.choice(ml_ds.users.ids(), size=500):
        ilist = recommend(pipe, uid)
        rl_topn = topn(items=ilist, n=20)
        rl_frac = samp_frac(items=ilist, n=20)
        assert len(rl_frac) == 20
        rl_one = samp_one(items=ilist, n=20)
        assert len(rl_one) == 20
        rl_hundred = samp_hundred(items=ilist, n=20)
        assert len(rl_hundred) == 20

        jc_frac.append(_jaccard(rl_topn, rl_frac))
        jc_one.append(_jaccard(rl_topn, rl_one))
        jc_hundred.append(_jaccard(rl_topn, rl_hundred))

    # high-temp should agree less than flat
    assert np.mean(jc_frac) < np.mean(jc_one)
    # low-temp should agree more than flat
    assert np.mean(jc_hundred) > np.mean(jc_one)


def _jaccard(il1: ItemList, il2: ItemList) -> float:
    s1 = set(il1.ids())
    s2 = set(il2.ids())

    num = len(s1 & s2)
    denom = len(s1 | s2)

    return num / denom
