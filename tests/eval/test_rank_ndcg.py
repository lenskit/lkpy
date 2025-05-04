# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

import hypothesis.strategies as st
from hypothesis import given
from pytest import approx, mark

from lenskit.data import ItemList
from lenskit.metrics import call_metric
from lenskit.metrics.ranking import NDCG
from lenskit.metrics.ranking._dcg import array_dcg, fixed_dcg


def test_dcg_empty():
    "empty should be zero"
    assert array_dcg(np.array([])) == approx(0)


def test_dcg_zeros():
    assert array_dcg(np.zeros(10)) == approx(0)


def test_dcg_single():
    "a single element should be scored at the right place"
    assert array_dcg(np.array([0.5])) == approx(0.5)
    assert array_dcg(np.array([0, 0.5])) == approx(0.5)
    assert array_dcg(np.array([0, 0, 0.5])) == approx(0.5 / np.log2(3))
    assert array_dcg(np.array([0, 0, 0.5, 0])) == approx(0.5 / np.log2(3))


def test_dcg_mult():
    "multiple elements should score correctly"
    assert array_dcg(np.array([np.e, np.pi])) == approx(np.e + np.pi)
    assert array_dcg(np.array([np.e, 0, 0, np.pi])) == approx(np.e + np.pi / np.log2(4))


def test_dcg_empty2():
    "empty should be zero"
    assert array_dcg(np.array([])) == approx(0)


def test_dcg_zeros2():
    assert array_dcg(np.zeros(10)) == approx(0)


def test_dcg_single2():
    "a single element should be scored at the right place"
    assert array_dcg(np.array([0.5])) == approx(0.5)
    assert array_dcg(np.array([0, 0.5])) == approx(0.5)
    assert array_dcg(np.array([0, 0, 0.5])) == approx(0.5 / np.log2(3))
    assert array_dcg(np.array([0, 0, 0.5, 0])) == approx(0.5 / np.log2(3))


def test_dcg_nan():
    "NANs should be 0"
    assert array_dcg(np.array([np.nan, 0.5])) == approx(0.5)


def test_dcg_mult2():
    "multiple elements should score correctly"
    assert array_dcg(np.array([np.e, np.pi])) == approx(np.e + np.pi)
    assert array_dcg(np.array([np.e, 0, 0, np.pi])) == approx((np.e + np.pi / np.log2(4)))


def test_ndcg_empty():
    recs = ItemList(ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(NDCG, recs, truth) == approx(0.0)


def test_ndcg_no_match():
    recs = ItemList([4], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(NDCG, recs, truth) == approx(0.0)


def test_ndcg_perfect():
    recs = ItemList([2, 3, 1], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(NDCG, recs, truth) == approx(1.0)


def test_ndcg_perfect_k_short():
    recs = ItemList([2, 3, 1], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(NDCG, recs, truth, k=2) == approx(1.0)
    assert call_metric(NDCG, recs[:2], truth, k=2) == approx(1.0)


def test_ndcg_shorter_not_best():
    recs = ItemList([1, 2], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    b_ideal = fixed_dcg(3)
    r_ideal = array_dcg(np.array([5.0, 4.0, 3.0]))
    assert call_metric(NDCG, recs, truth) == approx(fixed_dcg(2) / b_ideal)
    assert call_metric(NDCG, recs, truth, k=2) == approx(1.0)
    assert call_metric(NDCG, recs, truth, gain="rating") == approx(
        array_dcg(np.array([3.0, 5.0])) / r_ideal
    )


def test_ndcg_perfect_k():
    recs = ItemList([2, 3], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(NDCG, recs, truth, k=2) == approx(1.0)


def test_ndcg_perfect_k_norate():
    recs = ItemList([1, 3], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(NDCG, recs, truth, k=2) == approx(1.0)


def test_ndcg_almost_perfect_k_gain():
    recs = ItemList([1, 3], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(NDCG, recs, truth, k=2, gain="rating") == approx(
        array_dcg(np.array([3.0, 4.0])) / array_dcg(np.array([5.0, 4.0]))
    )


@given(
    st.lists(st.integers(1), min_size=1, max_size=100, unique=True),
    st.integers(1, 100),
)
def test_ndcg_alt_discount(items, k):
    rng = np.random.default_rng()
    picked = rng.choice(items, size=max(len(items) // 2, 1), replace=False)
    recs = ItemList(items, ordered=True)
    truth = ItemList(picked)

    mv_weighted = call_metric(NDCG, recs, truth, k=k)
    mv_legacy = call_metric(NDCG, recs, truth, k=k, discount=np.log2)

    try:
        assert mv_weighted == approx(mv_legacy)
    except Exception as e:
        e.add_note(f"recs: {recs}")
        e.add_note(f"truth: {truth}")
        raise e
