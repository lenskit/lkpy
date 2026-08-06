# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

import hypothesis.strategies as st
from hypothesis import given
from pytest import approx, raises

from lenskit.data import ItemList, ItemListCollection, from_interactions_df
from lenskit.metrics import MeasurementCollector
from lenskit.metrics.ranking import RBP, LogRankWeight
from lenskit.metrics.ranking._ips import IPSRBP
from lenskit.metrics.ranking._propensity import (
    FieldPropensity,
    PopularityPropensity,
    UniformPropensity,
    estimate_power_law_gamma,
)
from lenskit.testing import integer_ids

# item 101 has 4 users, 102 has 2, 103 has 1
POP_RATINGS = pd.DataFrame(
    {
        "user_id": [1, 2, 3, 4, 1, 2, 1],
        "item_id": [101, 101, 101, 101, 102, 102, 103],
    }
)
# same shape, with item IDs equal to their popularity
FIT_RATINGS = POP_RATINGS.replace({"item_id": {101: 4, 102: 2, 103: 1}})


def pop_data():
    return from_interactions_df(POP_RATINGS)


def fit_data():
    return from_interactions_df(FIT_RATINGS)


def data_with_counts(counts: dict[int, int]):
    "Build a dataset whose items have exactly the requested user counts."
    rows = [(user, item) for item, n in counts.items() for user in range(n)]
    return from_interactions_df(pd.DataFrame(rows, columns=["user_id", "item_id"]))


def mnar_log(alpha, seed=42, n_users=300, n_items=60, set_size=8):
    """
    Simulate a log in which popular items are more likely to be recorded.

    Users prefer popular items, and an interaction is recorded with probability
    proportional to a power ``alpha`` of item popularity.  ``alpha=0`` records
    every item at the same rate, so the log is missing-at-random.

    Returns recommendations from a popularity-chasing recommender, the complete
    preference sets, the recorded subset of them, and the log as a dataset.
    """
    rng = np.random.default_rng(seed)

    affinity = 1 / np.arange(1, n_items + 1)
    affinity /= affinity.sum()
    logits = np.log(affinity) + rng.gumbel(size=(n_users, n_items))
    prefs = np.argsort(-logits, axis=1)[:, :set_size]

    counts = np.maximum(np.bincount(prefs.ravel(), minlength=n_items), 1)
    popularity = counts / counts.max()
    recorded = rng.random((n_users, set_size)) < 0.5 * popularity[prefs] ** alpha

    relevant = np.zeros((n_users, n_items), dtype=bool)
    relevant[np.arange(n_users)[:, None], prefs] = True
    ranking = np.argsort(-(2.0 * relevant + popularity), axis=1)[:, :20]

    users = [u for u in range(n_users) if recorded[u].any()]
    rows = [(u, int(i)) for u in users for i in prefs[u][recorded[u]]]

    return (
        ItemListCollection.from_dict(
            {u: ItemList(ranking[u], ordered=True) for u in users}, key="user_id"
        ),
        ItemListCollection.from_dict({u: ItemList(prefs[u]) for u in users}, key="user_id"),
        ItemListCollection.from_dict(
            {u: ItemList(prefs[u][recorded[u]]) for u in users}, key="user_id"
        ),
        from_interactions_df(pd.DataFrame(rows, columns=["user_id", "item_id"])),
    )


def mean_score(metric, recs, truth):
    mc = MeasurementCollector()
    mc.add_metric(metric)
    summary, _ = mc.measure_run(recs, truth)
    return summary[f"{metric.label}.mean"]


@given(
    st.lists(integer_ids(), min_size=2, max_size=100, unique=True),
    st.floats(0.05, 0.95),
)
def test_uniform_unnormalized_matches_rbp(items, p):
    recs = ItemList(items, ordered=True)
    truth = ItemList(items[::2])

    ips = IPSRBP(propensity=UniformPropensity(), patience=p, self_normalized=False)
    assert ips.measure_list(recs, truth) == approx(RBP(patience=p).measure_list(recs, truth))


@given(
    st.lists(integer_ids(), min_size=2, max_size=100, unique=True),
    st.floats(0.05, 0.95),
    st.integers(1, 100),
)
def test_uniform_unnormalized_matches_rbp_truncated(items, p, n):
    recs = ItemList(items, ordered=True)
    truth = ItemList(items[::2])

    ips = IPSRBP(n, propensity=UniformPropensity(), patience=p, self_normalized=False)
    assert ips.measure_list(recs, truth) == approx(RBP(n, patience=p).measure_list(recs, truth))


@given(st.lists(integer_ids(), min_size=2, max_size=50, unique=True))
def test_uniform_unnormalized_matches_rbp_log_weight(items):
    recs = ItemList(items, ordered=True)
    truth = ItemList(items[::2])
    weight = LogRankWeight(offset=1)

    ips = IPSRBP(propensity=UniformPropensity(), weight=weight, self_normalized=False)
    assert ips.measure_list(recs, truth) == approx(RBP(weight=weight).measure_list(recs, truth))


@given(
    st.lists(integer_ids(), min_size=2, max_size=100, unique=True),
    st.floats(0.05, 0.95),
)
def test_uniform_self_normalized_is_aoa(items, p):
    "Uniform propensities give the average-over-all evaluator: RBP over |S*|."
    recs = ItemList(items, ordered=True)
    truth = ItemList(items[::2])

    ips = IPSRBP(propensity=UniformPropensity(), patience=p)
    expected = RBP(patience=p).measure_list(recs, truth) / len(truth)
    assert ips.measure_list(recs, truth) == approx(expected)


@given(st.floats(0.01, 100.0))
def test_self_normalized_is_scale_invariant(scale):
    recs = ItemList([1, 2, 3, 4], ordered=True)
    base = ItemList([1, 3, 5], propensity=[0.5, 0.25, 0.125])
    scaled = ItemList([1, 3, 5], propensity=np.array([0.5, 0.25, 0.125]) / scale)

    metric = IPSRBP(propensity=FieldPropensity())
    assert metric.measure_list(recs, scaled) == approx(metric.measure_list(recs, base))


def test_unnormalized_follows_propensity_scale():
    recs = ItemList([1, 2, 3, 4], ordered=True)
    base = ItemList([1, 3, 5], propensity=[0.5, 0.25, 0.125])
    scaled = ItemList([1, 3, 5], propensity=[0.05, 0.025, 0.0125])

    metric = IPSRBP(propensity=FieldPropensity(), self_normalized=False)
    assert metric.measure_list(recs, scaled) == approx(10 * metric.measure_list(recs, base))


def test_golden_self_normalized():
    # gamma 1 gives exponent 1, so with n*max = 4 the weights are 1, 2, 4; item
    # 103 hits at rank 1 and 101 at rank 3, patience 0.5:
    # (4 * 0.5 * 0.5**0 + 1 * 0.5 * 0.5**2) / (1 + 2 + 4)
    prop = PopularityPropensity(pop_data(), gamma=1.0)
    recs = ItemList([103, 999, 101], ordered=True)
    truth = ItemList([101, 102, 103])

    assert IPSRBP(propensity=prop, patience=0.5).measure_list(recs, truth) == approx(2.125 / 7.0)


def test_golden_unnormalized():
    prop = PopularityPropensity(pop_data(), gamma=1.0)
    recs = ItemList([103, 999, 101], ordered=True)
    truth = ItemList([101, 102, 103])

    metric = IPSRBP(propensity=prop, patience=0.5, self_normalized=False)
    assert metric.measure_list(recs, truth) == approx(2.125)


def test_denominator_is_not_truncated():
    prop = PopularityPropensity(pop_data(), gamma=1.0)
    recs = ItemList([103, 999, 101], ordered=True)
    truth = ItemList([101, 102, 103])

    # only the rank-1 hit counts, but the full weight sum still normalizes it
    metric = IPSRBP(1, propensity=prop, patience=0.5)
    assert metric.measure_list(recs, truth) == approx(2.0 / 7.0)


def test_serving_the_tail_scores_higher():
    prop = PopularityPropensity(pop_data(), gamma=1.0)
    truth = ItemList([101, 103])
    metric = IPSRBP(propensity=prop)

    tail_first = metric.measure_list(ItemList([103, 101], ordered=True), truth)
    popular_first = metric.measure_list(ItemList([101, 103], ordered=True), truth)
    assert tail_first > popular_first


def test_flat_popularity_leaves_the_score_alone():
    "With no popularity skew there is nothing to correct, for any gamma."
    data = data_with_counts({1: 5, 2: 5, 3: 5})
    recs = ItemList([2, 1], ordered=True)
    truth = ItemList([1, 2, 3])

    expected = IPSRBP(propensity=UniformPropensity()).measure_list(recs, truth)
    for gamma in [0.0, 1.0, 3.0]:
        prop = PopularityPropensity(data, gamma=gamma)
        assert IPSRBP(propensity=prop).measure_list(recs, truth) == approx(expected)


def test_serving_the_tail_gains_as_gamma_rises():
    data = data_with_counts({1: 16, 2: 4, 3: 1})
    truth = ItemList([1, 2, 3])
    recs = ItemList([3], ordered=True)

    scores = [
        IPSRBP(propensity=PopularityPropensity(data, gamma=g)).measure_list(recs, truth)
        for g in [-1.0, 0.0, 1.0, 2.0, 4.0]
    ]
    assert np.all(np.diff(scores) > 0)


def test_serving_the_head_loses_as_gamma_rises():
    data = data_with_counts({1: 16, 2: 4, 3: 1})
    truth = ItemList([1, 2, 3])
    recs = ItemList([1], ordered=True)

    scores = [
        IPSRBP(propensity=PopularityPropensity(data, gamma=g)).measure_list(recs, truth)
        for g in [-1.0, 0.0, 1.0, 2.0, 4.0]
    ]
    assert np.all(np.diff(scores) < 0)


def test_stronger_skew_amplifies_the_correction():
    "The more popularity-biased the log, the further the correction moves the score."
    truth = ItemList([1, 2, 3])
    recs = ItemList([1], ordered=True)
    aoa = IPSRBP(propensity=UniformPropensity()).measure_list(recs, truth)

    ratios = []
    for skew in [(4, 3, 2), (16, 4, 1), (64, 8, 1)]:
        prop = PopularityPropensity(data_with_counts(dict(zip([1, 2, 3], skew))), gamma=1.0)
        ratios.append(IPSRBP(propensity=prop).measure_list(recs, truth) / aoa)

    assert np.all(np.diff(ratios) < 0)


def test_clipping_pulls_back_toward_the_uncorrected_score():
    data = data_with_counts({1: 64, 2: 8, 3: 1})
    truth = ItemList([1, 2, 3])
    recs = ItemList([1], ordered=True)

    scores = [
        IPSRBP(propensity=PopularityPropensity(data, gamma=3.0, max_weight=cap)).measure_list(
            recs, truth
        )
        for cap in [None, 1000.0, 100.0, 10.0]
    ]
    aoa = IPSRBP(propensity=UniformPropensity()).measure_list(recs, truth)
    assert np.all(np.diff(scores) > 0)
    assert scores[-1] < aoa


def test_correction_beats_aoa_on_a_biased_log():
    "On a popularity-biased log, the correction is closer to the true reward."
    recs, complete, recorded, log = mnar_log(alpha=1.0)

    uniform = IPSRBP(10, propensity=UniformPropensity())
    true = mean_score(uniform, recs, complete)
    aoa = mean_score(uniform, recs, recorded)
    corrected = mean_score(
        IPSRBP(10, propensity=PopularityPropensity(log, gamma=1.0)), recs, recorded
    )

    assert aoa > true
    assert abs(corrected - true) < abs(aoa - true)


def test_correction_does_not_help_an_unbiased_log():
    "With a missing-at-random log there is nothing to correct."
    recs, complete, recorded, log = mnar_log(alpha=0.0)

    uniform = IPSRBP(10, propensity=UniformPropensity())
    true = mean_score(uniform, recs, complete)
    aoa = mean_score(uniform, recs, recorded)
    corrected = mean_score(
        IPSRBP(10, propensity=PopularityPropensity(log, gamma=1.0)), recs, recorded
    )

    assert aoa == approx(true, rel=0.05)
    assert abs(corrected - true) > abs(aoa - true)


def test_empty_test_list():
    metric = IPSRBP(propensity=UniformPropensity())
    assert np.isnan(metric.measure_list(ItemList([1, 2], ordered=True), ItemList([])))


def test_empty_recs():
    metric = IPSRBP(propensity=UniformPropensity())
    assert metric.measure_list(ItemList([], ordered=True), ItemList([1, 2, 3])) == approx(0.0)


def test_no_hits():
    metric = IPSRBP(propensity=UniformPropensity())
    assert metric.measure_list(ItemList([7, 8], ordered=True), ItemList([1, 2, 3])) == approx(0.0)


def test_unordered_recs_rejected():
    metric = IPSRBP(propensity=UniformPropensity())
    with raises(TypeError):
        metric.measure_list(ItemList([1, 2]), ItemList([1, 2, 3]))


def test_label():
    assert IPSRBP(propensity=UniformPropensity()).label == "IPSRBP"
    assert IPSRBP(10, propensity=UniformPropensity()).label == "IPSRBP@10"


def test_popularity_propensity_formula():
    prop = PopularityPropensity(pop_data(), gamma=2.0)
    assert prop.exponent == approx(1.5)

    ps = prop.propensities(ItemList([101, 102, 103]))
    assert ps == approx([1.0, (2 / 4) ** 1.5, (1 / 4) ** 1.5])


def test_popularity_propensity_uniform_at_gamma_minus_one():
    prop = PopularityPropensity(pop_data(), gamma=-1.0)
    assert prop.propensities(ItemList([101, 102, 103])) == approx([1.0, 1.0, 1.0])


def test_popularity_propensity_counts_interactions():
    prop = PopularityPropensity(pop_data(), gamma=1.0, count="interactions")
    assert prop.propensities(ItemList([101, 103])) == approx([1.0, 0.25])


def test_popularity_propensity_unknown_item():
    "Items absent from the source data count as one interaction."
    prop = PopularityPropensity(pop_data(), gamma=1.0)
    assert prop.propensities(ItemList([999])) == approx([0.25])
    assert prop.weights(ItemList([999])) == approx([4.0])


def test_popularity_propensity_max_weight():
    prop = PopularityPropensity(pop_data(), gamma=1.0, max_weight=2.0)
    assert prop.weights(ItemList([101, 102, 103])) == approx([1.0, 2.0, 2.0])
    assert prop.weights(ItemList([999])) == approx([2.0])


def test_popularity_propensity_clip_quantile():
    prop = PopularityPropensity(pop_data(), gamma=1.0, clip_quantile=0.5)
    assert prop.weights(ItemList([101, 102, 103])) == approx([1.0, 2.0, 2.0])


def test_popularity_propensity_empty_list():
    prop = PopularityPropensity(pop_data(), gamma=1.0)
    assert len(prop.propensities(ItemList([]))) == 0


def test_popularity_propensity_rejects_bad_params():
    data = pop_data()
    with raises(ValueError):
        PopularityPropensity(data, gamma=-2.0)
    with raises(ValueError):
        PopularityPropensity(data, gamma=1.0, max_weight=0.5)
    with raises(ValueError):
        PopularityPropensity(data, gamma=1.0, clip_quantile=1.5)


def test_uniform_propensity():
    prop = UniformPropensity()
    assert prop.propensities(ItemList([1, 2, 3])) == approx([1.0, 1.0, 1.0])
    assert prop.weights(ItemList([1, 2, 3])) == approx([1.0, 1.0, 1.0])


def test_field_propensity():
    items = ItemList([1, 2], propensity=[0.5, 0.25])
    assert FieldPropensity().weights(items) == approx([2.0, 4.0])


def test_field_propensity_missing_field():
    with raises(KeyError):
        FieldPropensity().propensities(ItemList([1, 2]))


def test_field_propensity_rejects_zero():
    with raises(ValueError):
        FieldPropensity().propensities(ItemList([1, 2], propensity=[0.5, 0.0]))


def test_estimate_gamma_recovers_slope():
    # items of popularity 1, 2 and 4 are recommended 1, 2 and 4 times, so f is
    # proportional to popularity and the fitted exponent is 1
    recs = ItemListCollection.from_dict(
        {
            1: ItemList([1, 2, 4], ordered=True),
            2: ItemList([2, 4], ordered=True),
            3: ItemList([4], ordered=True),
            4: ItemList([4], ordered=True),
        },
        key="user_id",
    )

    assert estimate_power_law_gamma(recs, fit_data(), exclude_quantile=0.0) == approx(1.0)


def test_estimate_gamma_steeper_bias():
    lists = {u: ItemList([4], ordered=True) for u in range(1, 15)}
    lists[15] = ItemList([1, 2, 4], ordered=True)
    lists[16] = ItemList([2, 4], ordered=True)
    recs = ItemListCollection.from_dict(lists, key="user_id")

    # 16, 2 and 1 recommendations at popularity 4, 2 and 1: a slope of 2
    assert estimate_power_law_gamma(recs, fit_data(), exclude_quantile=0.0) == approx(2.0)


def test_estimate_gamma_feeds_propensity_model():
    data = fit_data()
    recs = ItemListCollection.from_dict(
        {
            1: ItemList([1, 2, 4], ordered=True),
            2: ItemList([2, 4], ordered=True),
            3: ItemList([4], ordered=True),
            4: ItemList([4], ordered=True),
        },
        key="user_id",
    )

    gamma = estimate_power_law_gamma(recs, data, exclude_quantile=0.0)
    prop = PopularityPropensity(data, gamma=gamma)
    assert prop.propensities(ItemList([4, 2, 1])) == approx([1.0, 0.5, 0.25])


def test_estimate_gamma_needs_recommendations():
    with raises(ValueError):
        estimate_power_law_gamma(ItemListCollection.from_dict({}, key="user_id"), pop_data())
