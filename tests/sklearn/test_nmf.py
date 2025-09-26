# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import pandas as pd

from pytest import importorskip, mark

from lenskit.data import ItemList, from_interactions_df
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
    config = nmf.NMFConfig(max_iter=10, n_components=25)
    expected_ndcg = 0.22


@mark.parametrize("method", ["full", "minibatch"])
def test_nmf_basic_build(method):
    algo = nmf.NMFScorer(n_components=2, method=method)
    algo.train(simple_ds)

    assert algo.user_components.shape == (3, 2)


@mark.parametrize("method", ["full", "minibatch"])
def test_nmf_predict_basic(method):
    algo = nmf.NMFScorer(n_components=2, method=method)
    algo.train(simple_ds)

    preds = algo(10, ItemList([3]))
    preds = preds.scores("pandas", index="ids")
    assert preds.index[0] == 3
    assert 0 <= preds.loc[3] <= 5


@mark.parametrize("method", ["full", "minibatch"])
def test_nmf_predict_bad_item(method):
    algo = nmf.NMFScorer(n_components=2, method=method)
    algo.train(simple_ds)

    preds = algo(10, ItemList([4]))
    preds = preds.scores("pandas", index="ids")
    assert np.isnan(preds.loc[4])


@mark.parametrize("method", ["full", "minibatch"])
def test_nmf_predict_bad_user(method):
    algo = nmf.NMFScorer(n_components=2, method=method)
    algo.train(simple_ds)

    preds = algo(50, ItemList([3]))
    preds = preds.scores("pandas", index="ids")
    assert np.isnan(preds.loc[3])
