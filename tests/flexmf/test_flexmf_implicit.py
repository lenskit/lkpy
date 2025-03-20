# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle

import numpy as np
import pandas as pd
import torch

from pytest import approx, mark

from lenskit.data import Dataset, ItemList, RecQuery, from_interactions_df, load_movielens_df
from lenskit.flexmf import FlexMFImplicitConfig, FlexMFImplicitScorer
from lenskit.metrics import quick_measure_model
from lenskit.testing import BasicComponentTests, ScorerTests, wantjit


class TestFlexMFImplicit(BasicComponentTests, ScorerTests):
    component = FlexMFImplicitScorer
    config = FlexMFImplicitConfig(reg_method="AdamW", loss="pairwise")


@mark.slow
@mark.eval
def test_flexmf_test_accuracy(ml_100k):
    ds = from_interactions_df(ml_100k)
    results = quick_measure_model(
        FlexMFImplicitScorer(embedding_size=25, epochs=10, batch_size=1024),
        ds,
    )

    print(results.list_summary())

    assert results.global_metrics()["RBP"] >= 0.1
    assert results.global_metrics()["RBP"] < 0.2
