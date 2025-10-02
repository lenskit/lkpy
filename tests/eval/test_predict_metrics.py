# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import pandas as pd

from pytest import approx, mark, raises

from lenskit.data import ItemList, from_interactions_df
from lenskit.metrics import RunAnalysis, call_metric
from lenskit.metrics.predict import MAE, RMSE

_log = logging.getLogger(__name__)


def test_rmse_one():
    rmse = call_metric(RMSE, ItemList(["a"], scores=[1]), ItemList(["a"], rating=[1]))
    assert isinstance(rmse, float)
    assert rmse == approx(0)

    rmse = call_metric(RMSE, ItemList(["a"], scores=[1]), ItemList(["a"], rating=[2]))
    assert rmse == approx(1)

    rmse = call_metric(RMSE, ItemList(["a"], scores=[1]), ItemList(["a"], rating=[0.5]))
    assert rmse == approx(0.5)


def test_rmse_two():
    rmse = call_metric(
        RMSE, ItemList(["a", "b"], scores=[1, 2]), ItemList(["a", "b"], rating=[1, 2])
    )
    assert isinstance(rmse, float)
    assert rmse == approx(0)

    rmse = call_metric(
        RMSE, ItemList(["a", "b"], scores=[1, 1]), ItemList(["a", "b"], rating=[2, 2])
    )
    assert rmse == approx(1)

    rmse = call_metric(
        RMSE, ItemList(["a", "b"], scores=[1, 3]), ItemList(["a", "b"], rating=[3, 1])
    )
    assert rmse == approx(2)


def test_rmse_series_subset_items():
    rmse = call_metric(
        RMSE,
        ItemList(scores=[1, 3], item_ids=["a", "c"]),
        ItemList(rating=[3, 4, 1], item_ids=["a", "b", "c"]),
        missing_scores="ignore",
    )
    assert rmse == approx(2)


def test_rmse_series_missing_value_error():
    with raises(ValueError):
        call_metric(
            RMSE,
            ItemList(scores=[1, 3], item_ids=["a", "d"]),
            ItemList(rating=[3, 4, 1], item_ids=["a", "b", "c"]),
        )


def test_rmse_series_missing_value_ignore():
    rmse = call_metric(
        RMSE,
        ItemList(scores=[1, 3], item_ids=["a", "d"]),
        ItemList(rating=[3, 4, 1], item_ids=["a", "b", "c"]),
        missing_scores="ignore",
        missing_truth="ignore",
    )
    assert rmse == approx(2)


def test_mae_two():
    mae = call_metric(MAE, ItemList(["a", "b"], scores=[1, 2]), ItemList(["a", "b"], rating=[1, 2]))
    assert isinstance(mae, float)
    assert mae == approx(0)

    mae = call_metric(MAE, ItemList(["a", "b"], scores=[1, 1]), ItemList(["a", "b"], rating=[2, 2]))
    assert mae == approx(1)

    mae = call_metric(MAE, ItemList(["a", "b"], scores=[1, 3]), ItemList(["a", "b"], rating=[3, 1]))
    assert mae == approx(2)

    mae = call_metric(MAE, ItemList(["a", "b"], scores=[1, 3]), ItemList(["a", "b"], rating=[3, 2]))
    assert mae == approx(1.5)


@mark.slow
@mark.eval
def test_batch_rmse(ml_100k):
    from lenskit.basic import BiasScorer
    from lenskit.batch import predict
    from lenskit.pipeline import topn_pipeline
    from lenskit.splitting import SampleN, sample_users

    ds = from_interactions_df(ml_100k)

    bias = BiasScorer(damping=5)
    pipe = topn_pipeline(bias, predicts_ratings=True)

    split = sample_users(ds, 200, SampleN(5))
    pipe.train(split.train)

    preds = predict(pipe, split.test, n_jobs=1)

    pa = RunAnalysis()
    pa.add_metric(RMSE())
    pa.add_metric(MAE())

    metrics = pa.measure(preds, split.test)

    umdf = metrics.list_metrics(fill_missing=False)
    mdf = metrics.list_summary()

    # we should have all users
    assert len(umdf) == len(split.test)

    # we should only have users who are in the test data
    missing = set(umdf.index.tolist()) - set(k.user_id for k in split.test.keys())
    assert len(missing) == 0

    # we should not have any missing values
    assert all(umdf["RMSE"].notna())
    assert all(umdf["MAE"].notna())

    gs = metrics.global_metrics()
    _log.info("list metrics:\n%s", mdf)
    _log.info("global metrics:\n%s", gs)

    # we should have a reasonable mean
    assert umdf["RMSE"].mean() == approx(0.93, abs=0.05)
    assert mdf.loc["RMSE", "mean"] == approx(0.93, abs=0.05)

    assert umdf["MAE"].mean() == approx(0.76, abs=0.05)
    assert mdf.loc["MAE", "mean"] == approx(0.76, abs=0.05)

    # we should have global metrics
    assert gs["RMSE"] == approx(0.93, abs=0.05)
    assert gs["MAE"] == approx(0.76, abs=0.05)
