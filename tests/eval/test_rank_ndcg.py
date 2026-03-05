# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

import hypothesis.strategies as st
from hypothesis import given
from pytest import approx, mark, warns

from lenskit.data import ItemList
from lenskit.metrics import call_metric
from lenskit.metrics.ranking import NDCG
from lenskit.metrics.ranking._dcg import array_dcg, fixed_dcg
from lenskit.testing import integer_ids


def test_ndcg_empty():
    recs = ItemList(ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(NDCG, recs, truth) == approx(0.0)


def test_ndcg_empty_truth_returns_nan():
    recs = ItemList([1, 2, 3], ordered=True)
    truth = ItemList(ordered=True)
    val = call_metric(NDCG, recs, truth)
    assert np.isnan(val)


def test_ndcg_no_match():
    recs = ItemList([4], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(NDCG, recs, truth) == approx(0.0)


def test_ndcg_perfect():
    recs = ItemList([2, 3, 1], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(NDCG, recs, truth) == approx(1.0)


def test_ndcg_perfect_n_short():
    recs = ItemList([2, 3, 1], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(NDCG, recs, truth, n=2) == approx(1.0)
    assert call_metric(NDCG, recs[:2], truth, n=2) == approx(1.0)


def test_ndcg_perfect_k_short():
    recs = ItemList([2, 3, 1], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    with warns(DeprecationWarning):
        assert call_metric(NDCG, recs, truth, k=2) == approx(1.0)
    with warns(DeprecationWarning):
        assert call_metric(NDCG, recs[:2], truth, k=2) == approx(1.0)


def test_ndcg_shorter_not_best():
    recs = ItemList([1, 2], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    b_ideal = fixed_dcg(3)
    r_ideal = array_dcg(np.array([5.0, 4.0, 3.0]))
    assert call_metric(NDCG, recs, truth) == approx(fixed_dcg(2) / b_ideal)
    assert call_metric(NDCG, recs, truth, n=2) == approx(1.0)
    assert call_metric(NDCG, recs, truth, gain="rating") == approx(
        array_dcg(np.array([3.0, 5.0])) / r_ideal
    )


def test_ndcg_perfect_k():
    recs = ItemList([2, 3], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(NDCG, recs, truth, n=2) == approx(1.0)


def test_ndcg_perfect_k_norate():
    recs = ItemList([1, 3], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(NDCG, recs, truth, n=2) == approx(1.0)


def test_ndcg_almost_perfect_k_gain():
    recs = ItemList([1, 3], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(NDCG, recs, truth, n=2, gain="rating") == approx(
        array_dcg(np.array([3.0, 4.0])) / array_dcg(np.array([5.0, 4.0]))
    )


@given(
    st.lists(integer_ids(), min_size=1, max_size=100, unique=True),
    st.integers(1, 100),
)
def test_ndcg_alt_discount(items, n):
    rng = np.random.default_rng()
    picked = rng.choice(items, size=max(len(items) // 2, 1), replace=False)
    recs = ItemList(items, ordered=True)
    truth = ItemList(picked)

    mv_weighted = call_metric(NDCG, recs, truth, n=n)
    mv_legacy = call_metric(NDCG, recs, truth, n=n, discount=np.log2)

    try:
        assert mv_weighted == approx(mv_legacy)
    except Exception as e:
        e.add_note(f"recs: {recs}")
        e.add_note(f"truth: {truth}")
        raise e
