# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle
from itertools import product

import numpy as np
import pandas as pd
import torch

from pytest import approx, mark

from lenskit.data import Dataset, ItemList, RecQuery, from_interactions_df, load_movielens_df
from lenskit.flexmf import FlexMFImplicitConfig, FlexMFImplicitScorer
from lenskit.graphs.lightgcn import LightGCNConfig, LightGCNScorer
from lenskit.metrics import quick_measure_model
from lenskit.testing import BasicComponentTests, ScorerTests


class TestLightGCNPairwise(BasicComponentTests, ScorerTests):
    component = LightGCNScorer
    config = LightGCNConfig(loss="pairwise")


class TestLightGCNLogistic(BasicComponentTests, ScorerTests):
    component = LightGCNScorer
    config = LightGCNConfig(loss="logistic")


@mark.slow
@mark.eval
def test_lightgcn_test_accuracy(ml_100k):
    ds = from_interactions_df(ml_100k)
    results = quick_measure_model(
        LightGCNScorer(embedding_size=25, epochs=10, batch_size=1024),
        ds,
    )

    print(results.list_summary())

    assert results.list_summary().loc["RBP", "mean"] >= 0.01
    assert results.list_summary().loc["RBP", "mean"] < 0.25
