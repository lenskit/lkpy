# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import pandas as pd

import hypothesis.strategies as st
from hypothesis import given
from pytest import approx, mark

from lenskit.data import ItemList
from lenskit.metrics import call_metric
from lenskit.metrics.ranking import RBP, LogRankWeight
from lenskit.metrics.ranking._rbp import rank_biased_precision
from lenskit.testing import demo_recs, integer_ids  # noqa: F401

_log = logging.getLogger(__name__)


def test_rbp_empty():
    recs = ItemList([], ordered=True)
    truth = ItemList([1, 2, 3])
    assert call_metric(RBP, recs, truth) == approx(0.0)


def test_rbp_no_match():
    recs = ItemList([4], ordered=True)
    truth = ItemList([1, 2, 3])
    assert call_metric(RBP, recs, truth) == approx(0.0)


def test_rbp_one_match():
    recs = ItemList([1], ordered=True)
    truth = ItemList([1, 2, 3])
    assert call_metric(RBP, recs, truth) == approx(0.15)


@given(st.lists(integer_ids(), min_size=1, max_size=100, unique=True), st.floats(0.05, 0.95))
def test_rbp_perfect(items, p):
    n = len(items)
    recs = ItemList(items, ordered=True)
    truth = ItemList(items)
    assert call_metric(RBP, recs, truth, patience=p) == approx(np.sum(p ** np.arange(n)) * (1 - p))


@given(st.lists(integer_ids(), min_size=1, max_size=100, unique=True), st.floats(0.05, 0.95))
def test_rbp_perfect_norm(items, p):
    recs = ItemList(items, ordered=True)
    truth = ItemList(items)
    assert call_metric(RBP, recs, truth, patience=p, normalize=True) == approx(1.0)


@given(
    st.lists(integer_ids(), min_size=1, max_size=100, unique=True),
    st.integers(1, 100),
    st.floats(0.05, 0.95),
)
def test_rbp_perfect_k(items, k, p):
    n = len(items)
    eff_n = min(n, k)
    recs = ItemList(items, ordered=True)
    truth = ItemList(items)
    assert call_metric(RBP, recs, truth, k, patience=p) == approx(
        np.sum(p ** np.arange(eff_n)) * (1 - p)
    )


@given(
    st.lists(integer_ids(), min_size=1, max_size=100, unique=True),
    st.integers(1, 100),
)
def test_rbp_perfect_log_weight(items, k):
    recs = ItemList(items, ordered=True)
    truth = ItemList(items)
    assert call_metric(RBP, recs, truth, k, weight=LogRankWeight(offset=1)) == approx(1.0)


@given(
    st.lists(integer_ids(), min_size=2, max_size=100, unique=True),
)
def test_rbp_partial_log_weight(items):
    recs = ItemList(items, ordered=True)
    truth = ItemList(items[::2])
    w = np.reciprocal(np.log2(np.arange(1, len(recs) + 1) + 1))
    gw = np.zeros_like(w)
    gw[::2] = w[::2]
    assert call_metric(RBP, recs, truth, weight=LogRankWeight(offset=1)) == approx(
        np.sum(gw) / np.sum(w)
    )


@given(
    st.lists(integer_ids(), min_size=1, max_size=100, unique=True),
    st.integers(1, 100),
    st.floats(0.05, 0.95),
)
def test_rbp_perfect_k_norm(items, k, p):
    recs = ItemList(items, ordered=True)
    truth = ItemList(items)
    assert call_metric(RBP, recs, truth, k, patience=p, normalize=True) == approx(1.0)


def test_rbp_missing():
    recs = ItemList([1, 2], ordered=True)
    truth = ItemList([1, 2, 3])
    # (1 + 0.5) * 0.5
    assert call_metric(RBP(patience=0.5), recs, truth) == approx(0.75)


def test_rbp_weight_field():
    items = [1, 2, 3, 4, 5]
    weights = [1.0, 0.8, 0.6, 0.4, 0.2]

    recs = ItemList(item_ids=items, weight=weights, ordered=True)
    truth = ItemList([2, 4])  # items at positions 2 and 4

    # (0.8 + 0.4) / (1.0 + 0.8 + 0.6 + 0.4 + 0.2) = 1.2 / 3.0 = 0.4
    assert call_metric(RBP, recs, truth, weight_field="weight") == approx(0.4)


def test_rank_biased_precision():
    good = np.array([False, True, False, True, False])
    weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    result = rank_biased_precision(good, weights, normalization=3.0)
    assert result == approx(1.2 / 3.0)


# test for graded rbp


def test_rbp_empty_graded():
    recs = ItemList([], ordered=True)
    truth = ItemList(item_ids=[1, 2, 3], grade=[1.0, 1.0, 1.0])

    metric = RBP(grade_field="grade")
    assert metric.measure_list(recs, truth) == approx(0.0)


def test_rbp_unknown_grade_default():
    recs = ItemList([1, 2], ordered=True)
    truth = ItemList(item_ids=[1], grade=[1.0])

    p = 0.5
    metric = RBP(patience=p, grade_field="grade", unknown_grade=0.25)

    # RBP = (1-p)*(relevance[0] + relevance[1]*p)
    expected = (1 - p) * (1 + 0.25 * p)
    assert metric.measure_list(recs, truth) == approx(expected)


def test_rbp_unknown_grade():
    recs = ItemList([1, 2], ordered=True)
    truth = ItemList(item_ids=[1], grade=[1.0])

    p = 0.5
    metric = RBP(patience=p, grade_field="grade", unknown_grade=0.30)

    # RBP = (1-p)*(relevance[0] + relevance[1]*p)
    expected = (1 - p) * (1 + 0.30 * p)
    assert metric.measure_list(recs, truth) == approx(expected)


def test_rbp_binary_vs_graded_equivalent():
    recs = ItemList([1, 3], ordered=True)

    graded_truth = ItemList(item_ids=[1, 3], grade=[1.0, 1.0])
    binary_truth = ItemList([1, 3])  # no grade field

    grbp = RBP(grade_field="grade")
    rbp = RBP()  # binary

    assert grbp.measure_list(recs, graded_truth) == approx(rbp.measure_list(recs, binary_truth))
