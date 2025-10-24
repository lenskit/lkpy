# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np

import hypothesis.strategies as st
from hypothesis import given
from pytest import approx

from lenskit.data.accum import ValueStatAccumulator


def test_collect_empty():
    acc = ValueStatAccumulator()
    rv = acc.accumulate()

    assert rv["n"] == 0
    assert np.isnan(rv["mean"])
    assert np.isnan(rv["median"])
    assert np.isnan(rv["std"])


@given(st.floats(allow_infinity=False, allow_nan=False))
def test_collect_one(x):
    acc = ValueStatAccumulator()
    acc.add(x)
    rv = acc.accumulate()

    assert rv["n"] == 1
    assert rv["mean"] == x
    assert rv["median"] == x
    assert rv["std"] == approx(0.0)


@given(st.lists(st.floats(allow_infinity=False, allow_nan=False), min_size=2))
def test_collect_list(xs):
    acc = ValueStatAccumulator()
    for i, x in enumerate(xs):
        assert len(acc) == i
        acc.add(x)
    rv = acc.accumulate()

    assert rv["n"] == len(xs)
    assert rv["mean"] == approx(np.mean(xs))
    assert rv["median"] == approx(np.median(xs))
    assert rv["std"] == approx(np.std(xs), nan_ok=True)


@given(st.lists(st.floats(allow_infinity=False)))
def test_collect_list_nan(xs):
    acc = ValueStatAccumulator()
    for x in xs:
        acc.add(x)
    rv = acc.accumulate()

    xs = np.array(xs)
    finite = np.isfinite(xs)
    assert rv["n"] == np.sum(finite)
    assert rv["mean"] == approx(np.mean(xs[finite]), nan_ok=True)
    assert rv["median"] == approx(np.median(xs[finite]), nan_ok=True)
    assert rv["std"] == approx(np.std(xs[finite]), nan_ok=True)


@given(st.lists(st.floats(allow_infinity=False)), st.lists(st.floats(allow_infinity=False)))
def test_split_combine(xs, ys):
    acc = ValueStatAccumulator()
    right = acc.split_accumulator()

    for x in xs:
        acc.add(x)

    for y in ys:
        right.add(y)

    acc.combine(right)
    rv = acc.accumulate()

    xs = np.array(xs + ys)
    finite = np.isfinite(xs)
    assert rv["n"] == np.sum(finite)
    assert rv["mean"] == approx(np.mean(xs[finite]), nan_ok=True)
    assert rv["median"] == approx(np.median(xs[finite]), nan_ok=True)
    assert rv["std"] == approx(np.std(xs[finite]), nan_ok=True)
