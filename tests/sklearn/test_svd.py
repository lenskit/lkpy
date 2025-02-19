# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle

import numpy as np
import pandas as pd

from pytest import approx, mark

from lenskit.data import Dataset, ItemList, from_interactions_df
from lenskit.metrics import call_metric, quick_measure_model
from lenskit.sklearn import svd
from lenskit.testing import BasicComponentTests, ScorerTests

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)
simple_ds = from_interactions_df(simple_df)

need_skl = mark.skipif(not svd.SKL_AVAILABLE, reason="scikit-learn not installed")


class TestBiasedSVD(BasicComponentTests, ScorerTests):
    component = svd.BiasedSVDScorer


@need_skl
def test_svd_basic_build():
    algo = svd.BiasedSVDScorer(features=2)
    algo.train(simple_ds)

    assert algo.user_components_.shape == (3, 2)


@need_skl
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


@need_skl
def test_svd_predict_bad_item():
    algo = svd.BiasedSVDScorer(features=2)
    algo.train(simple_ds)

    preds = algo(10, ItemList([4]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


@need_skl
def test_svd_predict_bad_user():
    algo = svd.BiasedSVDScorer(features=2)
    algo.train(simple_ds)

    preds = algo(50, ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


@need_skl
@mark.slow
def test_svd_save_load(ml_ds: Dataset):
    original = svd.BiasedSVDScorer(features=20)
    original.train(ml_ds)

    mod = pickle.dumps(original)
    _log.info("serialized to %d bytes", len(mod))
    algo = pickle.loads(mod)

    assert algo.bias_.global_bias == original.bias_.global_bias
    assert np.all(algo.bias_.user_biases == original.bias_.user_biases)
    assert np.all(algo.bias_.item_biases == original.bias_.item_biases)
    assert np.all(algo.user_components_ == original.user_components_)


@need_skl
@mark.slow
@mark.eval
def test_svd_batch_accuracy(rng, ml_100k: pd.DataFrame):
    data = from_interactions_df(ml_100k)
    svd_algo = svd.BiasedSVDScorer(features=25, damping=10)
    results = quick_measure_model(svd_algo, data, predicts_ratings=True, rng=rng)

    assert results.global_metrics()["MAE"] == approx(0.71, abs=0.025)
    assert results.list_summary().loc["RMSE", "mean"] == approx(0.92, abs=0.05)
