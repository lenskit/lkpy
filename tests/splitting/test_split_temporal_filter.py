# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import datetime as dt

import numpy as np

from pytest import approx

from lenskit.data import Dataset
from lenskit.splitting import split_global_time, split_temporal_fraction


def test_temporal_split_filtered_users(ml_ds: Dataset):
    point = dt.datetime.fromisoformat("2015-01-01")

    split = split_global_time(ml_ds, "2015-01-01", filter_test_users=True)
    assert np.all(split.train.interaction_table(format="pandas")["timestamp"] < point)

    train_users = split.train.user_stats().index[split.train.user_stats()["count"] > 0]

    for u, il in split.test:
        ts = il.field("timestamp", format="pandas")

        assert ts is not None
        assert np.all(ts >= point)
        assert u.user_id in train_users


def test_temporal_split_filtered_users_fraction(ml_ds: Dataset):
    split = split_temporal_fraction(ml_ds, 0.2, filter_test_users=True)

    point = min(il.field("timestamp").min() for il in split.test.lists())
    assert np.all(split.train.interaction_table(format="pandas")["timestamp"] < point)
