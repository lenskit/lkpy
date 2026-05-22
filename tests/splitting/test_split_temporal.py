# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import datetime as dt

import numpy as np

from pytest import approx

from lenskit.data import Dataset
from lenskit.splitting import split_global_time, split_temporal_fraction


def test_temporal_split(ml_ds: Dataset):
    point = dt.datetime.fromisoformat("2015-01-01")

    split = split_global_time(ml_ds, "2015-01-01")

    n_test = sum(len(il) for il in split.test.lists())
    assert n_test + split.train.interaction_count == ml_ds.interaction_count

    assert np.all(split.train.interaction_table(format="pandas")["timestamp"] < point)
    for u, il in split.test:
        ts = il.field("timestamp", format="pandas")
        assert ts is not None
        assert np.all(ts >= point)


def test_temporal_split_limit(ml_ds: Dataset):
    split = split_global_time(ml_ds, "2015-01-01", filter_test_users=True)

    tr_count = split.train.user_stats()

    for k in split.test.keys():
        (u,) = k
        assert tr_count.loc[u, "rating_count"] >= 1


def test_temporal_split_limit2(ml_ds: Dataset):
    split = split_global_time(ml_ds, "2015-01-01", filter_test_users=2)

    tr_count = split.train.user_stats()

    for k in split.test.keys():
        (u,) = k
        assert tr_count.loc[u, "rating_count"] >= 2


def test_temporal_split_fraction(ml_ds: Dataset):
    split = split_temporal_fraction(ml_ds, 0.2)

    n_test = sum(len(il) for il in split.test.lists())
    assert n_test + split.train.interaction_count == ml_ds.interaction_count
    assert n_test / ml_ds.interaction_count == approx(0.2, abs=0.01)

    point = min(il.field("timestamp").min() for il in split.test.lists())
    assert np.all(split.train.interaction_table(format="pandas")["timestamp"] < point)


def test_multi_split(ml_ds: Dataset):
    p1 = dt.datetime.fromisoformat("2015-01-01")
    p2 = dt.datetime.fromisoformat("2016-01-01")

    valid, test = split_global_time(ml_ds, ["2015-01-01", "2016-01-01"])

    n_test = sum(len(il) for il in test.test.lists())
    n_valid = sum(len(il) for il in valid.test.lists())
    assert n_test + test.train.interaction_count == ml_ds.interaction_count
    assert n_test + n_valid + valid.train.interaction_count == ml_ds.interaction_count

    assert np.all(valid.train.interaction_table(format="pandas")["timestamp"] < p1)
    assert np.all(test.train.interaction_table(format="pandas")["timestamp"] < p2)
    for u, il in test.test:
        ts = il.field("timestamp", format="pandas")
        assert ts is not None
        assert np.all(ts >= p2)

    for u, il in valid.test:
        ts = il.field("timestamp", format="pandas")
        assert ts is not None
        assert np.all(ts >= p1)
        assert np.all(ts < p2)
