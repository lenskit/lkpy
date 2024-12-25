# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
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
from lenskit.metrics.ranking import RBP
from lenskit.testing import demo_recs  # noqa: F401

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
    assert call_metric(RBP, recs, truth) == approx(0.5)


@given(st.lists(st.integers(1), min_size=1, max_size=100, unique=True), st.floats(0.05, 0.95))
def test_rbp_perfect(items, p):
    n = len(items)
    recs = ItemList(items, ordered=True)
    truth = ItemList(items)
    assert call_metric(RBP, recs, truth, patience=p) == approx(np.sum(p ** np.arange(n)) * (1 - p))


@given(st.lists(st.integers(1), min_size=1, max_size=100, unique=True), st.floats(0.05, 0.95))
def test_rbp_perfect_norm(items, p):
    recs = ItemList(items, ordered=True)
    truth = ItemList(items)
    assert call_metric(RBP, recs, truth, patience=p, normalize=True) == approx(1.0)


@given(
    st.lists(st.integers(1), min_size=1, max_size=100, unique=True),
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
    st.lists(st.integers(1), min_size=1, max_size=100, unique=True),
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
    assert call_metric(RBP, recs, truth) == approx(0.75)
