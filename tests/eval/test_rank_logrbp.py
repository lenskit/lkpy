# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import math

import numpy as np

import hypothesis.strategies as st
from hypothesis import given
from pytest import approx, fixture, raises

from lenskit.data import ItemList, ItemListCollection
from lenskit.metrics import RBP, LogRBP, MeasurementCollector, measure_list
from lenskit.metrics.ranking import GeometricRankWeight, LogRankWeight
from lenskit.metrics.ranking._logrbp import LogRBPAccumulator, log_rank_biased_precision
from lenskit.testing import demo_recs, integer_ids

_log = logging.getLogger(__name__)


@fixture
def two_users():
    """
    Two users with hand-checkable scores: A retrieves at ranks 1, 3 and 5; C
    retrieves none of its three test items (a real zero, not a missing value).
    """
    outputs = ItemListCollection.from_dict(
        {
            "A": ItemList([1, 2, 3, 4, 5], ordered=True),
            "C": ItemList([10, 11, 12, 13, 14], ordered=True),
        },
        key="user_id",
    )
    tests = ItemListCollection.from_dict(
        {
            "A": ItemList([1, 3, 5]),
            "C": ItemList([1, 2, 3]),
        },
        key="user_id",
    )
    return outputs, tests


def rbp_geo(patience: float, ranks: list[int]) -> float:
    """Independent reference: RBP = (1 - p) * sum(p^(r-1)) over retrieved ranks."""
    return (1 - patience) * sum(patience ** (r - 1) for r in ranks)


# --- per-list behaviour -------------------------------------------------------


def test_logrbp_empty_recs():
    recs = ItemList([], ordered=True)
    truth = ItemList([1, 2, 3])
    # RBP is 0 here, so log-RBP is -inf: a valid measurement, not a missing one.
    assert measure_list(LogRBP, recs, truth) == -np.inf


def test_logrbp_no_match():
    recs = ItemList([4], ordered=True)
    truth = ItemList([1, 2, 3])
    assert measure_list(LogRBP, recs, truth) == -np.inf


def test_logrbp_missing_truth():
    recs = ItemList([1, 2], ordered=True)
    # an empty test list has no defined score (same guard as RBP)
    val = measure_list(LogRBP, recs, ItemList([]))
    assert np.isnan(val)


def test_logrbp_one_match():
    recs = ItemList([1], ordered=True)
    truth = ItemList([1, 2, 3])
    assert measure_list(LogRBP, recs, truth, patience=0.5) == approx(math.log(0.5))


def test_logrbp_no_weight_arguments():
    """Custom weights are deliberately unsupported: see the class docstring."""
    recs = ItemList([1, 2, 3], ordered=True)
    truth = ItemList([1, 3])
    with raises(TypeError):
        measure_list(LogRBP, recs, truth, weight=LogRankWeight())
    with raises(TypeError):
        measure_list(LogRBP, recs, truth, weight_field="weight")


@given(
    st.lists(integer_ids(), min_size=1, max_size=100, unique=True),
    st.floats(0.05, 0.95),
)
def test_logrbp_perfect(items, p):
    n = len(items)
    recs = ItemList(items, ordered=True)
    truth = ItemList(items)
    # log of the closed form, computed without touching LensKit's RBP
    assert measure_list(LogRBP, recs, truth, patience=p) == approx(
        math.log((1 - p) * sum(p**i for i in range(n)))
    )


@given(
    st.lists(integer_ids(), min_size=1, max_size=100, unique=True),
    st.floats(0.05, 0.95),
)
def test_logrbp_perfect_norm(items, p):
    recs = ItemList(items, ordered=True)
    truth = ItemList(items)
    # a perfect list normalizes to RBP 1, whose log is exactly 0
    assert measure_list(LogRBP, recs, truth, patience=p, normalize=True) == approx(0.0)


@given(
    st.lists(integer_ids(), min_size=1, max_size=60, unique=True),
    st.integers(1, 60),
    st.floats(0.05, 0.95),
)
def test_logrbp_matches_rbp(items, n, p):
    """
    Mathematical equivalence gate: log-RBP must equal the log of ordinary RBP for
    every list where RBP is defined and non-zero.
    """
    recs = ItemList(items, ordered=True)
    truth = ItemList(items[::2])
    rbp = measure_list(RBP, recs, truth, n, patience=p)
    log_rbp = measure_list(LogRBP, recs, truth, n, patience=p)
    if rbp == 0.0:
        assert log_rbp == -np.inf
    elif np.isnan(rbp):
        assert np.isnan(log_rbp)
    else:
        assert log_rbp == approx(math.log(rbp), rel=1e-9)
        assert math.exp(log_rbp) == approx(rbp, rel=1e-9)


@given(
    st.lists(integer_ids(), min_size=1, max_size=60, unique=True),
    st.floats(0.05, 0.95),
)
def test_logrbp_matches_rbp_normalized(items, p):
    recs = ItemList(items, ordered=True)
    truth = ItemList(items[::3])
    rbp = measure_list(RBP, recs, truth, patience=p, normalize=True)
    log_rbp = measure_list(LogRBP, recs, truth, patience=p, normalize=True)
    assert log_rbp == approx(math.log(rbp), rel=1e-9)


# --- the deep-rank case that motivates log space ------------------------------


def test_logrbp_rank_5001_underflows_rbp_only():
    """
    A single relevant item at rank 5001 scores exactly 0.0 in normal space (float64
    subnormal floor) but stays finite in log space.
    """
    rank = 5001
    p = 0.85
    recs = ItemList(list(range(1, rank + 1)), ordered=True)
    truth = ItemList([rank])

    rbp = measure_list(RBP, recs, truth, patience=p)
    log_rbp = measure_list(LogRBP, recs, truth, patience=p)
    reference = math.log(1 - p) + (rank - 1) * math.log(p)

    assert rbp == 0.0  # underflowed to a hard zero
    assert np.isfinite(log_rbp)
    assert log_rbp == approx(reference, rel=1e-12)
    # the normal-space value cannot be recovered: log(0) is not merely "very small"
    with raises(ValueError):
        math.log(rbp)


def test_logrbp_rank_5001_aggregate_is_finite():
    """
    Three users whose only relevant item sits at rank 5001: log(mean(RBP)) is a
    finite number, whereas the normal-space pipeline yields mean 0 and cannot form
    the logarithm at all.
    """
    rank, p = 5001, 0.85
    recs = ItemList(list(range(1, rank + 1)), ordered=True)
    truth = ItemList([rank])

    mc = MeasurementCollector()
    mc.add_metric(LogRBP(patience=p))
    mc.add_metric(RBP(patience=p))
    for user in ["u1", "u2", "u3"]:
        mc.add_list_measurement(recs, truth, user=user)

    summ = mc.summary_metrics()
    assert summ["LogRBP.n"] == 3
    assert summ["RBP.mean"] == 0.0
    assert summ["LogRBP.logmean"] == approx(math.log(1 - p) + (rank - 1) * math.log(p), rel=1e-12)


def test_geometric_weight_never_exponentiated():
    """
    The per-list computation uses log weights directly; the weight itself is zero in
    normal space while the log weight is finite.
    """
    gw = GeometricRankWeight(0.85)
    ranks = np.array([5001], dtype=np.int32)
    assert gw.weight(ranks)[0] == 0.0
    assert np.isfinite(gw.log_weight(ranks)[0])
    good = np.zeros(1, dtype=bool)
    good[0] = True
    assert log_rank_biased_precision(
        good, gw.log_weight(ranks), math.log(gw.series_sum())
    ) == approx(gw.log_weight(ranks)[0] - math.log(gw.series_sum()))


# --- accumulator semantics ----------------------------------------------------


def test_accumulator_drops_missing_keeps_neginf():
    acc = LogRBPAccumulator()
    for v in [math.log(0.5), None, np.nan, -np.inf]:
        acc.add(v)
    assert len(acc) == 2  # the finite value and -inf; None and NaN excluded
    assert list(acc.values) == approx([math.log(0.5), -np.inf])


def test_accumulator_no_valid_measurements_is_not_zero():
    acc = LogRBPAccumulator()
    acc.add(np.nan)
    acc.add(None)
    rv = acc.accumulate()
    assert rv["n"] == 0
    assert np.isnan(rv["logmean"])  # undefined, not a fabricated zero


def test_accumulator_all_zero_rbp_aggregates_to_neginf():
    acc = LogRBPAccumulator()
    for _ in range(4):
        acc.add(-np.inf)
    rv = acc.accumulate()
    assert rv["n"] == 4
    assert rv["logmean"] == -np.inf


def test_accumulator_beyond_preallocated_capacity():
    # 2000 exceeds ValueStatAccumulator's 1024-slot pre-allocation, so the
    # delegated grow path is exercised even though the buffer is no longer local.
    acc = LogRBPAccumulator()
    n = 2000
    for _ in range(n):
        acc.add(math.log(0.9))
    assert len(acc) == n
    rv = acc.accumulate()
    assert rv["n"] == n
    assert rv["logmean"] == approx(math.log(0.9), rel=1e-12)


@given(st.floats(0.05, 0.999))
def test_logrbp_matches_rbp_across_patience(p):
    """
    Equivalence must hold at the patience extremes too: p near 1 makes the infinite
    series sum large, p small makes it approach 1.
    """
    recs = ItemList(list(range(1, 13)), ordered=True)
    truth = ItemList([1, 5, 12])
    rbp = measure_list(RBP, recs, truth, patience=p)
    log_rbp = measure_list(LogRBP, recs, truth, patience=p)
    assert log_rbp == approx(math.log(rbp), rel=1e-9)
    assert math.exp(log_rbp) == approx(rbp, rel=1e-9)


def test_logrbp_extreme_patience_values():
    """Hand-checked values at both patience ends, with normalize on and off."""
    recs = ItemList([1, 2, 3], ordered=True)
    truth = ItemList([1])
    # p -> small: only the first position matters, RBP = (1-p) * 1
    for p in (0.05, 0.001):
        assert measure_list(LogRBP, recs, truth, patience=p) == approx(math.log(1 - p), rel=1e-12)
    # p -> 1: the series sum blows up while the per-list log stays finite
    p = 0.999
    assert measure_list(LogRBP, recs, truth, patience=p) == approx(math.log(1 - p), rel=1e-9)
    assert measure_list(LogRBP, recs, truth, patience=p, normalize=True) == approx(0.0)


def test_accumulator_mixed_zero_and_nonzero():
    """
    A zero-valued RBP contributes e^{-inf} = 0 to the sum but still counts in the
    denominator, which is what separates log(mean) from mean(log).
    """
    rbps = [0.15, 0.007, 0.0, 0.0003, np.nan, None]
    # convert to log space the way the metric would: 0 -> -inf, undefined -> NaN
    logs = []
    for v in rbps:
        if v is None:
            logs.append(None)
        elif np.isnan(v):
            logs.append(np.nan)
        elif v == 0.0:
            logs.append(-np.inf)
        else:
            logs.append(math.log(v))

    acc = LogRBPAccumulator()
    for v in logs:
        acc.add(v)

    valid = [v for v in rbps if v is not None and not np.isnan(v)]
    rv = acc.accumulate()
    assert rv["n"] == len(valid) == 4
    assert rv["logmean"] == approx(math.log(sum(valid) / len(valid)), rel=1e-12)
    # contrast: the mean of logs is -inf for the same data
    log_vals = [l for l in logs if l is not None and not np.isnan(l)]
    assert np.mean(log_vals) == -np.inf


# --- collection-level behaviour ----------------------------------------------


def test_logrbp_summary_keys():
    mc = MeasurementCollector()
    mc.add_metric(LogRBP())
    mc.add_list_measurement(ItemList([1, 2], ordered=True), ItemList([1]), user="u")
    keys = set(mc.summary_metrics().keys())
    assert keys == {"LogRBP.n", "LogRBP.logmean"}
    # deliberately no mean/median/std of the log values
    assert "LogRBP.mean" not in keys
    assert "LogRBP.median" not in keys
    assert "LogRBP.std" not in keys


def test_logrbp_labels():
    assert LogRBP().label == "LogRBP"
    assert LogRBP(10).label == "LogRBP@10"
    assert LogRBP(n=5).label == "LogRBP@5"


def test_logrbp_mixed_zero_nonzero_users(two_users):
    outputs, tests = two_users
    mc = MeasurementCollector()
    mc.add_metric(LogRBP())
    mc.add_metric(RBP())
    mc.add_collection_measurements(outputs, tests)
    summ = mc.summary_metrics()

    assert summ["RBP.n"] == 2
    assert summ["LogRBP.n"] == 2
    assert summ["RBP.mean"] == approx(rbp_geo(0.85, [1, 3, 5]) / 2, rel=1e-12)
    assert summ["LogRBP.logmean"] == approx(math.log(summ["RBP.mean"]), rel=1e-9)


def test_logmean_exponentiated_equals_rbp_mean(two_users):
    """
    The headline equivalence: exp(logmean) must reproduce RBP's arithmetic mean
    exactly, including in the presence of a zero-scoring list.
    """
    outputs, tests = two_users
    mc = MeasurementCollector()
    mc.add_metric(LogRBP())
    mc.add_metric(RBP())
    mc.add_collection_measurements(outputs, tests)
    summ = mc.summary_metrics()

    assert math.exp(summ["LogRBP.logmean"]) == approx(summ["RBP.mean"], rel=1e-9)


def test_logrbp_missing_truth_excluded(two_users):
    """A user with no test items leaves both denominators, identically to RBP."""
    mc = MeasurementCollector()
    mc.add_metric(LogRBP())
    mc.add_metric(RBP())
    mc.add_list_measurement(ItemList([1, 2, 3], ordered=True), ItemList([1, 2, 3]), user="A")
    mc.add_list_measurement(ItemList([1, 2, 3], ordered=True), ItemList([]), user="E")
    summ = mc.summary_metrics()

    assert summ["RBP.n"] == 1
    assert summ["LogRBP.n"] == 1
    # user A is perfect: RBP normalizes by the infinite series, so log is not 0
    assert summ["LogRBP.logmean"] == approx(math.log(summ["RBP.mean"]), rel=1e-9)


def test_logrbp_empty_recs_normalize_returns_nan():
    """
    RBP raises ZeroDivisionError for this degenerate combination; LogRBP reports an
    undefined measurement instead.  Documented divergence pending #1159.
    """
    recs = ItemList([], ordered=True)
    truth = ItemList([1, 2, 3])
    assert np.isnan(measure_list(LogRBP, recs, truth, normalize=True))
    with raises(ZeroDivisionError):
        measure_list(RBP, recs, truth, normalize=True)


def test_logrbp_truncation_matches_rbp():
    items = list(range(1, 21))
    recs = ItemList(items, ordered=True)
    truth = ItemList(items)
    for n in (1, 3, 7, 20):
        rbp = measure_list(RBP, recs, truth, n)
        assert measure_list(LogRBP, recs, truth, n) == approx(math.log(rbp), rel=1e-9)


def test_logrbp_requires_ordered_for_truncation():
    with raises(ValueError, match="ordered"):
        measure_list(LogRBP(3), ItemList([1, 2, 3, 4], ordered=False), ItemList([1, 2]))


def test_logrbp_on_demo_recs(demo_recs):
    """
    End-to-end over real ML-latest-small derived data: the log aggregate must
    exponentiate back to RBP's arithmetic mean.
    """
    split, recs = demo_recs
    mc = MeasurementCollector()
    mc.add_metric(LogRBP())
    mc.add_metric(RBP())
    # RunMetrics exposes summary_metrics as a field, not a method
    summ = mc.measure_run(recs, split.test).summary_metrics

    _log.info("demo recs summary: %s", summ)
    assert summ["LogRBP.n"] == summ["RBP.n"]
    assert summ["LogRBP.n"] > 0
    assert math.exp(summ["LogRBP.logmean"]) == approx(summ["RBP.mean"], rel=1e-6)
