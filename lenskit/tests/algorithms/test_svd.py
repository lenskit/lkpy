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

from lenskit.algorithms import svd
from lenskit.data import Dataset, from_interactions_df
from lenskit.metrics import call_metric
from lenskit.util import clone

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)
simple_ds = from_interactions_df(simple_df)

need_skl = mark.skipif(not svd.SKL_AVAILABLE, reason="scikit-learn not installed")


@need_skl
def test_svd_basic_build():
    algo = svd.BiasedSVD(2)
    algo.fit(simple_ds)

    assert algo.user_components_.shape == (3, 2)


@need_skl
def test_svd_predict_basic():
    _log.info("SVD input data:\n%s", simple_df)
    algo = svd.BiasedSVD(2, damping=0)
    _log.info("SVD bias: %s", algo.bias)
    algo.fit(simple_ds)
    _log.info("user means:\n%s", str(algo.bias.user_offsets_))
    _log.info("item means:\n%s", str(algo.bias.item_offsets_))
    _log.info("user matrix:\n%s", str(algo.user_components_))

    preds = algo.predict_for_user(10, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= 0
    assert preds.loc[3] <= 5


@need_skl
def test_svd_predict_bad_item():
    algo = svd.BiasedSVD(2)
    algo.fit(simple_ds)

    preds = algo.predict_for_user(10, [4])
    assert len(preds) == 1
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


@need_skl
def test_svd_predict_bad_user():
    algo = svd.BiasedSVD(2)
    algo.fit(simple_ds)

    preds = algo.predict_for_user(50, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


@need_skl
def test_svd_clone():
    algo = svd.BiasedSVD(5, damping=10)

    a2 = clone(algo)
    assert a2.factorization.n_components == algo.factorization.n_components
    assert a2.bias.user_damping == algo.bias.user_damping
    assert a2.bias.item_damping == algo.bias.item_damping


@need_skl
@mark.slow
def test_svd_save_load(ml_ds: Dataset):
    original = svd.BiasedSVD(20)
    original.fit(ml_ds)

    mod = pickle.dumps(original)
    _log.info("serialized to %d bytes", len(mod))
    algo = pickle.loads(mod)

    assert algo.bias.mean_ == original.bias.mean_
    assert np.all(algo.bias.user_offsets_ == original.bias.user_offsets_)
    assert np.all(algo.bias.item_offsets_ == original.bias.item_offsets_)
    assert np.all(algo.user_components_ == original.user_components_)


@need_skl
@mark.slow
@mark.eval
def test_svd_batch_accuracy(ml_100k: pd.DataFrame):
    import lenskit.crossfold as xf
    import lenskit.metrics.predict as pm
    from lenskit import batch
    from lenskit.algorithms import basic, bias

    svd_algo = svd.BiasedSVD(25, damping=10)
    algo = basic.Fallback(svd_algo, bias.Bias(damping=10))

    def eval(train, test):
        _log.info("running training")
        algo.fit(from_interactions_df(train))
        _log.info("testing %d users", test.user.nunique())
        return batch.predict(algo, test)

    folds = xf.partition_users(ml_100k, 5, xf.SampleFrac(0.2))
    preds = pd.concat(eval(train, test) for (train, test) in folds)
    mae = call_metric(pm.MAE, preds)
    assert mae == approx(0.74, abs=0.025)

    user_rmse = pm.measure_user_predictions(preds, pm.RMSE)
    assert user_rmse.mean() == approx(0.92, abs=0.05)
