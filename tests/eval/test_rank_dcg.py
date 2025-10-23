# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from pytest import approx, mark

from lenskit.data import ItemList
from lenskit.metrics import call_metric
from lenskit.metrics.ranking import DCG
from lenskit.metrics.ranking._dcg import array_dcg, fixed_dcg


@mark.parametrize("method", ["graded", "binary"])
def test_array_dcg_empty(method):
    "empty should be zero"
    assert array_dcg(np.array([]), graded=method == "graded") == approx(0)


@mark.parametrize("method", ["graded", "binary"])
def test_array_dcg_zeros(method):
    assert array_dcg(np.zeros(10), graded=method == "graded") == approx(0)


@mark.parametrize("method", ["graded", "binary"])
def test_array_dcg_single(method):
    "a single element should be scored at the right place"
    val = 0.5 if method == "graded" else 1.0
    assert array_dcg(np.array([0.5]), graded=method == "graded") == approx(val)
    assert array_dcg(np.array([0, 0.5]), graded=method == "graded") == approx(val)
    assert array_dcg(np.array([0, 0, 0.5]), graded=method == "graded") == approx(val / np.log2(3))
    assert array_dcg(np.array([0, 0, 0.5, 0]), graded=method == "graded") == approx(
        val / np.log2(3)
    )


def test_array_dcg_mult():
    "multiple elements should score correctly"
    assert array_dcg(np.array([np.e, np.pi])) == approx(np.e + np.pi)
    assert array_dcg(np.array([np.e, 0, 0, np.pi])) == approx(np.e + np.pi / np.log2(4))


@mark.parametrize("method", ["graded", "binary"])
def test_array_dcg_empty2(method):
    "empty should be zero"
    assert array_dcg(np.array([]), graded=method == "graded") == approx(0)


@mark.parametrize("method", ["graded", "binary"])
def test_array_dcg_zeros2(method):
    assert array_dcg(np.zeros(10), graded=method == "graded") == approx(0)


def test_array_dcg_nan():
    "NANs should be 0"
    assert array_dcg(np.array([np.nan, 0.5])) == approx(0.5)


def test_dcg_empty():
    recs = ItemList(ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(DCG, recs, truth) == approx(0.0)


def test_dcg_no_match():
    recs = ItemList([4], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(DCG, recs, truth) == approx(0.0)


def test_dcg_perfect():
    recs = ItemList([2, 3, 1], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(DCG, recs, truth) == approx(np.sum(np.reciprocal(np.log2([2, 2, 3]))))


def test_dcg_perfect_k_short():
    recs = ItemList([2, 3, 1], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(DCG, recs, truth, n=2) == approx(2.0)
    assert call_metric(DCG, recs[:2], truth, n=2) == approx(2.0)


def test_dcg_shorter_not_best():
    recs = ItemList([1, 2], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(DCG, recs, truth) == approx(fixed_dcg(2))
    assert call_metric(DCG, recs, truth, n=2) == approx(2.0)
    assert call_metric(DCG, recs, truth, gain="rating") == approx(array_dcg(np.array([3.0, 5.0])))


def test_dcg_perfect_k():
    recs = ItemList([2, 3], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(DCG, recs, truth, n=2) == approx(2.0)


def test_dcg_perfect_k_norate():
    recs = ItemList([1, 3], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(DCG, recs, truth, n=2) == approx(2.0)


def test_dcg_almost_perfect_k_gain():
    recs = ItemList([1, 3], ordered=True)
    truth = ItemList([1, 2, 3], rating=[3.0, 5.0, 4.0])
    assert call_metric(DCG, recs, truth, n=2, gain="rating") == approx(
        array_dcg(np.array([3.0, 4.0]))
    )
