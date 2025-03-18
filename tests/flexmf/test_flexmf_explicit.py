import logging
import pickle

import numpy as np
import pandas as pd
import torch

from pytest import approx, mark

from lenskit.data import Dataset, ItemList, RecQuery, from_interactions_df, load_movielens_df
from lenskit.flexmf import FlexMFExplicitScorer
from lenskit.metrics import quick_measure_model
from lenskit.testing import BasicComponentTests, ScorerTests, wantjit


class TestFlexMFExplicit(BasicComponentTests, ScorerTests):
    component = FlexMFExplicitScorer


@mark.slow
@mark.eval
def test_flexmf_test_accuracy(ml_100k):
    ds = from_interactions_df(ml_100k)
    results = quick_measure_model(
        FlexMFExplicitScorer(embedding_size=25, epochs=10, reg=0.1, learning_rate=0.005),
        ds,
        predicts_ratings=True,
    )

    print(results.list_summary())

    assert results.global_metrics()["MAE"] == approx(0.73, abs=0.045)
    assert results.list_summary().loc["RMSE", "mean"] == approx(0.94, abs=0.05)
