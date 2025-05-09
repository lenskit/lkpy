# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle

import numpy as np
import pandas as pd

from pytest import importorskip, mark

from lenskit.data import Dataset, ItemList, from_interactions_df
from lenskit.metrics import quick_measure_model
from lenskit.testing import BasicComponentTests, ScorerTests

nmf = importorskip("lenskit.sklearn.nmf")

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)
simple_ds = from_interactions_df(simple_df)


class TestNMF(BasicComponentTests, ScorerTests):
    component = nmf.NMFScorer
    config = nmf.NMFConfig(max_iter=10, n_components=5)


def test_nmf_basic_build():
    algo = nmf.NMFScorer(n_components=2)
    algo.train(simple_ds)

    assert algo.user_components.shape == (3, 2)


def test_nmf_predict_basic():
    _log.info("NMF input data:\n%s", simple_df)
    algo = nmf.NMFScorer(n_components=2)
    algo.train(simple_ds)
    _log.info("user matrix:\n%s", str(algo.user_components))
    _log.info("item matrix:\n%s", str(algo.item_components))

    preds = algo(10, ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert preds.loc[3] >= 0
    assert preds.loc[3] <= 5


def test_nmf_predict_bad_item():
    algo = nmf.NMFScorer(n_components=2)
    algo.train(simple_ds)

    preds = algo(10, ItemList([4]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_nmf_predict_bad_user():
    algo = nmf.NMFScorer(n_components=2)
    algo.train(simple_ds)

    preds = algo(50, ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


@mark.slow
@mark.eval
def test_nmf_batch_accuracy(rng, ml_100k: pd.DataFrame):
    data = from_interactions_df(ml_100k)
    svd_algo = nmf.NMFScorer(n_components=25)
    results = quick_measure_model(svd_algo, data, predicts_ratings=True, rng=rng)

    ndcg = results.list_summary().loc["NDCG", "mean"]
    _log.info("nDCG for users is %.4f", ndcg)
    assert ndcg > 0.22  # type: ignore
