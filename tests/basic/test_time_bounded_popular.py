# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from lenskit.basic import popularity
from lenskit.data import from_interactions_df
from lenskit.data.items import ItemList
from lenskit.testing import ScorerTests

ts = datetime(year=2024, month=1, day=1)
one_day_ago = ts - timedelta(days=1)
two_days_ago = ts - timedelta(days=2)
simple_df = pd.DataFrame(
    {
        "item_id": [1, 2, 2, 3],
        "user_id": [10, 12, 10, 13],
        "rating": [4.0, 3.0, 5.0, 2.0],
        "timestamp": [ts, one_day_ago, one_day_ago, one_day_ago],
    }
)
simple_ds = from_interactions_df(simple_df)


class TestTimeBoundedPop(ScorerTests):
    component = popularity.TimeBoundedPopScore
    config = popularity.TimeBoundedPopConfig(cutoff="2015-01-01")

    def verify_models_equivalent(self, orig, copy):
        assert all(orig.item_scores_ == copy.item_scores_)


def test_time_bounded_pop_score_quantile_one_day_window():
    algo = popularity.TimeBoundedPopScore(cutoff=one_day_ago)
    algo.train(simple_ds)
    assert np.all(algo.item_scores_ == [1.0, 0.0, 0.0])


def test_time_bounded_pop_score_quantile_one_day_window_call_interface():
    algo = popularity.TimeBoundedPopScore(cutoff=one_day_ago)
    algo.train(simple_ds)
    p = algo(ItemList(item_ids=[1, 2, 3]))

    assert len(p) == 3
    assert (p.scores() == [1.0, 0.0, 0.0]).all()


def test_time_bounded_pop_score_quantile_two_day_window():
    algo = popularity.TimeBoundedPopScore(cutoff=two_days_ago)
    algo.train(simple_ds)
    assert np.all(algo.item_scores_ == pd.Series([0.25, 1.0, 0.5], index=[1, 2, 3]))


def test_time_bounded_pop_score_fallbacks_to_pop_score_for_dataset_without_timestamps():
    ds = from_interactions_df(simple_df.drop(columns=["timestamp"]))

    algo = popularity.TimeBoundedPopScore(cutoff=one_day_ago)
    algo.train(ds)
    assert np.all(algo.item_scores_ == pd.Series([0.25, 1.0, 0.5], index=[1, 2, 3]))


def test_time_bounded_pop_score_rank():
    algo = popularity.TimeBoundedPopScore(cutoff=two_days_ago, score="rank")
    algo.train(simple_ds)
    assert np.all(algo.item_scores_ == pd.Series([1.5, 3.0, 1.5], index=[1, 2, 3]))


def test_time_bounded_pop_score_counts():
    algo = popularity.TimeBoundedPopScore(cutoff=two_days_ago, score="count")
    algo.train(simple_ds)
    assert np.all(algo.item_scores_ == pd.Series([1, 2, 1], index=[1, 2, 3], dtype=np.int32))
