# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle

import numpy as np
import pandas as pd

from pytest import approx, importorskip, mark

from lenskit.data import Dataset, ItemList, from_interactions_df
from lenskit.metrics import quick_measure_model
from lenskit.testing import BasicComponentTests, ScorerTests

svd = importorskip("lenskit.sklearn.svd")

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)
simple_ds = from_interactions_df(simple_df)


class TestBiasedSVD(BasicComponentTests, ScorerTests):
    component = svd.BiasedSVDScorer
    config = svd.BiasedSVDConfig(embedding_size=25, damping=10)
    expected_rmse = (0.915, 0.925)

    def verify_models_equivalent(self, orig, copy):
        assert copy.bias_.global_bias == orig.bias_.global_bias
        assert np.all(copy.bias_.user_biases == orig.bias_.user_biases)
        assert np.all(copy.bias_.item_biases == orig.bias_.item_biases)
        assert np.all(copy.user_components_ == orig.user_components_)


def test_svd_basic_build():
    algo = svd.BiasedSVDScorer(features=2)
    algo.train(simple_ds)

    assert algo.user_components_.shape == (3, 2)


def test_svd_predict_basic():
    _log.info("SVD input data:\n%s", simple_df)
    algo = svd.BiasedSVDScorer(features=2, damping=0)
    algo.train(simple_ds)
    _log.info("user means:\n%s", str(algo.bias_.user_biases))
    _log.info("item means:\n%s", str(algo.bias_.item_biases))
    _log.info("user matrix:\n%s", str(algo.user_components_))

    preds = algo(10, ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert preds.loc[3] >= 0
    assert preds.loc[3] <= 5


def test_svd_predict_bad_item():
    algo = svd.BiasedSVDScorer(features=2)
    algo.train(simple_ds)

    preds = algo(10, ItemList([4]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_svd_predict_bad_user():
    algo = svd.BiasedSVDScorer(features=2)
    algo.train(simple_ds)

    preds = algo(50, ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])
