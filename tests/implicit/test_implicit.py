# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle

import numpy as np

from pytest import approx, importorskip, mark

from lenskit.data import ItemList, from_interactions_df
from lenskit.metrics import quick_measure_model
from lenskit.testing import BasicComponentTests, ScorerTests

imp = importorskip("lenskit.implicit")
_log = logging.getLogger(__name__)


class TestImplicitALS(BasicComponentTests, ScorerTests):
    component = imp.ALS
    expected_ndcg = 0.05


class TestImplicitBPR(BasicComponentTests, ScorerTests):
    component = imp.BPR
    expected_ndcg = 0.05


@mark.slow
def test_implicit_als_train_rec(ml_ds):
    algo = imp.ALS(factors=25)
    assert algo.config.factors == 25

    ret = algo.train(ml_ds)
    assert ret is algo

    preds = algo(100, items=ItemList([1, 377]))
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert np.all(preds.notna())

    _log.info("serializing implicit model")
    mod = pickle.dumps(algo)
    _log.info("serialized to %d bytes")
    a2 = pickle.loads(mod)

    p2 = a2(100, items=ItemList([1, 377])).scores("pandas", index="ids")
    assert np.all(p2 == preds)


@mark.slow
def test_implicit_bpr_train_rec(ml_ds):
    algo = imp.BPR(factors=25, use_gpu=False)
    assert algo.config.factors == 25

    algo.train(ml_ds)

    preds = algo(100, ItemList([20, 30, 23148010]))
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert all(preds.index == [20, 30, 23148010])
    assert all(preds.isna() == [False, False, True])

    _log.info("serializing implicit model")
    mod = pickle.dumps(algo)
    _log.info("serialized to %d bytes")
    a2 = pickle.loads(mod)

    p2 = a2(100, items=ItemList([20, 30, 23148010])).scores("pandas", index="ids")
    assert p2.values == approx(preds.values, nan_ok=True)


def test_implicit_pickle_untrained(tmp_path):
    mf = tmp_path / "bpr.dat"
    algo = imp.BPR(factors=25, use_gpu=False)

    with mf.open("wb") as f:
        pickle.dump(algo, f)

    with mf.open("rb") as f:
        a2 = pickle.load(f)

    assert a2 is not algo
    assert a2.config.factors == 25
