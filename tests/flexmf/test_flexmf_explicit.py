# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle

import numpy as np
import pandas as pd
import torch

from pytest import approx, mark

from lenskit.data import Dataset, ItemList, RecQuery, from_interactions_df, load_movielens_df
from lenskit.flexmf import FlexMFExplicitScorer
from lenskit.flexmf._explicit import FlexMFExplicitConfig
from lenskit.metrics import quick_measure_model
from lenskit.testing import BasicComponentTests, ScorerTests, wantjit


class TestFlexMFExplicit(BasicComponentTests, ScorerTests):
    component = FlexMFExplicitScorer


class TestFlexMFExplicitAdam(BasicComponentTests, ScorerTests):
    component = FlexMFExplicitScorer
    config = FlexMFExplicitConfig(reg_method="AdamW")


@mark.slow
@mark.eval
def test_flexmf_test_accuracy(ml_100k):
    ds = from_interactions_df(ml_100k)
    results = quick_measure_model(
        FlexMFExplicitScorer(embedding_size=25, epochs=10, batch_size=1024),
        ds,
        predicts_ratings=True,
    )

    summary = results.list_summary()
    gs = results.global_metrics()

    print(summary)

    assert gs["MAE"] == approx(0.26, abs=0.05)
    assert summary.loc["RMSE", "mean"] == approx(0.36, abs=0.05)
