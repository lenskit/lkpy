# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from datetime import datetime, timedelta
import pickle

import numpy as np
import pandas as pd

from lenskit.lenskit.data.convert import from_interactions_df
from lenskit.algorithms import basic

day = timedelta(days=1)
now = int(datetime.now().timestamp())
one_day_ago = now - day.total_seconds()
simple_df = pd.DataFrame(
    {
        "item": [1, 2, 2, 3],
        "user": [10, 12, 10, 13],
        "rating": [4.0, 3.0, 5.0, 2.0],
        "timestamp": [now, one_day_ago, one_day_ago, one_day_ago],
    }
)
simple_ds = from_interactions_df(simple_df)


def test_time_bounded_pop_score_quantile_one_day_window():
    algo = basic.TimeBoundedPopScore(day)
    algo.fit(simple_ds)
    assert algo.item_scores_.equals(pd.Series([1.0, 0.0, 0.0], index=[1, 2, 3]))


def test_time_bounded_pop_score_quantile_two_day_window():
    algo = basic.TimeBoundedPopScore(2 * day)
    algo.fit(simple_ds)
    assert algo.item_scores_.equals(pd.Series([0.25, 1.0, 0.5], index=[1, 2, 3]))


def test_time_bounded_pop_score_rank():
    algo = basic.TimeBoundedPopScore(2 * day, "rank")
    algo.fit(simple_ds)
    assert algo.item_scores_.equals(pd.Series([1.5, 3.0, 1.5], index=[1, 2, 3]))


def test_time_bounded_pop_score_counts(rng):
    algo = basic.TimeBoundedPopScore(2 * day, "count")
    algo.fit(simple_ds)
    assert algo.item_scores_.equals(pd.Series([1, 2, 1], index=[1, 2, 3], dtype=np.int32))


def test_time_bounded_pop_score_save_load():
    original = basic.TimeBoundedPopScore(day)
    original.fit(simple_ds)

    mod = pickle.dumps(original)
    algo = pickle.loads(mod)

    assert all(algo.item_scores_ == original.item_scores_)
