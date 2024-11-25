# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from lenskit.algorithms import basic
from lenskit.data import from_interactions_df

ts = datetime(year=2024, month=1, day=1)
one_day_ago = ts - timedelta(days=1)
two_days_ago = ts - timedelta(days=2)
simple_df = pd.DataFrame(
    {
        "item": [1, 2, 2, 3],
        "user": [10, 12, 10, 13],
        "rating": [4.0, 3.0, 5.0, 2.0],
        "timestamp": [i.timestamp() for i in [ts, one_day_ago, one_day_ago, one_day_ago]],
    }
)
simple_ds = from_interactions_df(simple_df)


def test_time_bounded_pop_score_quantile_one_day_window():
    algo = basic.TimeBoundedPopScore(one_day_ago)
    algo.fit(simple_ds)
    assert algo.item_scores_.equals(pd.Series([1.0, 0.0, 0.0], index=[1, 2, 3]))


def test_time_bounded_pop_score_quantile_two_day_window():
    algo = basic.TimeBoundedPopScore(two_days_ago)
    algo.fit(simple_ds)
    assert algo.item_scores_.equals(pd.Series([0.25, 1.0, 0.5], index=[1, 2, 3]))


def test_time_bounded_pop_score_fallbacks_to_pop_score_for_dataset_without_timestamps():
    ds = from_interactions_df(simple_df.drop(columns=["timestamp"]))

    algo = basic.TimeBoundedPopScore(one_day_ago)
    algo.fit(ds)
    assert algo.item_scores_.equals(pd.Series([0.25, 1.0, 0.5], index=[1, 2, 3]))


def test_time_bounded_pop_score_rank():
    algo = basic.TimeBoundedPopScore(two_days_ago, "rank")
    algo.fit(simple_ds)
    assert algo.item_scores_.equals(pd.Series([1.5, 3.0, 1.5], index=[1, 2, 3]))


def test_time_bounded_pop_score_counts():
    algo = basic.TimeBoundedPopScore(two_days_ago, "count")
    algo.fit(simple_ds)
    assert algo.item_scores_.equals(pd.Series([1, 2, 1], index=[1, 2, 3], dtype=np.int32))


def test_time_bounded_pop_score_save_load():
    original = basic.TimeBoundedPopScore(one_day_ago)
    original.fit(simple_ds)

    mod = pickle.dumps(original)
    algo = pickle.loads(mod)

    assert all(algo.item_scores_ == original.item_scores_)
