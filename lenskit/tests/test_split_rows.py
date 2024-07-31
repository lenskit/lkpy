# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import functools as ft
import itertools as it
import math

import numpy as np
import pandas as pd

import pytest

import lenskit.crossfold as xf


def test_partition_rows(ml_ratings: pd.DataFrame):
    splits = xf.partition_rows(ml_ratings, 5)
    splits = list(splits)
    assert len(splits) == 5

    for s in splits:
        assert len(s.test) + len(s.train) == len(ml_ratings)
        assert all(s.test.index.union(s.train.index) == ml_ratings.index)
        test_idx = s.test.set_index(["user", "item"]).index
        train_idx = s.train.set_index(["user", "item"]).index
        assert len(test_idx.intersection(train_idx)) == 0

    # we should partition!
    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue

        i1 = s1.test.set_index(["user", "item"]).index
        i2 = s2.test.set_index(["user", "item"]).index
        inter = i1.intersection(i2)
        assert len(inter) == 0

    union = ft.reduce(lambda i1, i2: i1.union(i2), (s.test.index for s in splits))
    assert len(union.unique()) == len(ml_ratings)


def test_sample_rows(ml_ratings: pd.DataFrame):
    splits = xf.sample_rows(ml_ratings, partitions=5, size=1000)
    splits = list(splits)
    assert len(splits) == 5

    for s in splits:
        assert len(s.test) == 1000
        assert len(s.test) + len(s.train) == len(ml_ratings)
        test_idx = s.test.set_index(["user", "item"]).index
        train_idx = s.train.set_index(["user", "item"]).index
        assert len(test_idx.intersection(train_idx)) == 0

    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue

        i1 = s1.test.set_index(["user", "item"]).index
        i2 = s2.test.set_index(["user", "item"]).index
        inter = i1.intersection(i2)
        assert len(inter) == 0


def test_sample_rows_more_smaller_parts(ml_ratings: pd.DataFrame):
    splits = xf.sample_rows(ml_ratings, partitions=10, size=500)
    splits = list(splits)
    assert len(splits) == 10

    for s in splits:
        assert len(s.test) == 500
        assert len(s.test) + len(s.train) == len(ml_ratings)
        test_idx = s.test.set_index(["user", "item"]).index
        train_idx = s.train.set_index(["user", "item"]).index
        assert len(test_idx.intersection(train_idx)) == 0

    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue

        i1 = s1.test.set_index(["user", "item"]).index
        i2 = s2.test.set_index(["user", "item"]).index
        inter = i1.intersection(i2)
        assert len(inter) == 0


def test_sample_non_disjoint(ml_ratings: pd.DataFrame):
    splits = xf.sample_rows(ml_ratings, partitions=10, size=1000, disjoint=False)
    splits = list(splits)
    assert len(splits) == 10

    for s in splits:
        assert len(s.test) == 1000
        assert len(s.test) + len(s.train) == len(ml_ratings)
        test_idx = s.test.set_index(["user", "item"]).index
        train_idx = s.train.set_index(["user", "item"]).index
        assert len(test_idx.intersection(train_idx)) == 0

    # There are enough splits & items we should pick at least one duplicate
    ipairs = (
        (s1.test.set_index(["user", "item"]).index, s2.test.set_index(["user", "item"]).index)
        for (s1, s2) in it.product(splits, splits)
    )
    isizes = [len(i1.intersection(i2)) for (i1, i2) in ipairs]
    assert any(n > 0 for n in isizes)


@pytest.mark.slow
def test_sample_oversize(ml_ratings: pd.DataFrame):
    splits = xf.sample_rows(ml_ratings, 50, 10000)
    splits = list(splits)
    assert len(splits) == 50

    for s in splits:
        assert len(s.test) + len(s.train) == len(ml_ratings)
        assert all(s.test.index.union(s.train.index) == ml_ratings.index)
        test_idx = s.test.set_index(["user", "item"]).index
        train_idx = s.train.set_index(["user", "item"]).index
        assert len(test_idx.intersection(train_idx)) == 0
