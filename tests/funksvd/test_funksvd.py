# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from pytest import approx, importorskip, mark

from lenskit import batch
from lenskit.data import Dataset, ItemList, ItemListCollection, UserIDKey, from_interactions_df
from lenskit.pipeline.common import predict_pipeline
from lenskit.testing import BasicComponentTests, ScorerTests

funk = importorskip("lenskit.funksvd")

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)
simple_ds = from_interactions_df(simple_df)


class TestFunkSVD(BasicComponentTests, ScorerTests):
    component = funk.FunkSVDScorer
    expected_rmse = (0.87, 0.97)

    def verify_models_equivalent(self, orig, copy):
        assert copy.bias.global_bias == orig.bias.global_bias
        assert np.all(copy.bias.user_biases == orig.bias.user_biases)
        assert np.all(copy.bias.item_biases == orig.bias.item_biases)
        assert np.all(copy.user_embeddings == orig.user_embeddings)
        assert np.all(copy.item_embeddings == orig.item_embeddings)
        assert np.all(copy.items.index == orig.items.index)
        assert np.all(copy.users.index == orig.users.index)


def test_fsvd_basic_build():
    algo = funk.FunkSVDScorer(features=20, epochs=20)
    assert algo.config is not None
    assert algo.config.embedding_size == 20
    algo.train(simple_ds)

    assert algo.bias is not None
    assert algo.bias.global_bias == approx(simple_df.rating.mean())
    assert algo.item_embeddings.shape == (3, 20)
    assert algo.user_embeddings.shape == (3, 20)


def test_fsvd_clamp_build():
    algo = funk.FunkSVDScorer(features=20, epochs=20, range=(1, 5))
    algo.train(simple_ds)

    assert algo.bias is not None
    assert algo.bias.global_bias == approx(simple_df.rating.mean())
    assert algo.item_embeddings.shape == (3, 20)
    assert algo.user_embeddings.shape == (3, 20)


def test_fsvd_predict_basic():
    algo = funk.FunkSVDScorer(features=20, epochs=20)
    algo.train(simple_ds)

    assert algo.bias is not None
    assert algo.bias.global_bias == approx(simple_df.rating.mean())
    assert algo.item_embeddings.shape == (3, 20)
    assert algo.user_embeddings.shape == (3, 20)

    preds = algo(query=10, items=ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert preds.loc[3] >= 0
    assert preds.loc[3] <= 5


def test_fsvd_predict_clamp():
    algo = funk.FunkSVDScorer(features=20, epochs=20, range=(1, 5))
    algo.train(simple_ds)

    assert algo.bias is not None
    assert algo.bias.global_bias == approx(simple_df.rating.mean())
    assert algo.item_embeddings.shape == (3, 20)
    assert algo.user_embeddings.shape == (3, 20)

    preds = algo(query=10, items=ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert preds.loc[3] >= 1
    assert preds.loc[3] <= 5


def test_fsvd_predict_bad_item():
    algo = funk.FunkSVDScorer(features=20, epochs=20)
    algo.train(simple_ds)

    assert algo.bias is not None
    assert algo.bias.global_bias == approx(simple_df.rating.mean())
    assert algo.item_embeddings.shape == (3, 20)
    assert algo.user_embeddings.shape == (3, 20)

    preds = algo(10, ItemList([4]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_fsvd_predict_bad_item_clamp():
    algo = funk.FunkSVDScorer(features=20, epochs=20, range=(1, 5))
    algo.train(simple_ds)

    assert algo.bias is not None
    assert algo.bias.global_bias == approx(simple_df.rating.mean())
    assert algo.item_embeddings.shape == (3, 20)
    assert algo.user_embeddings.shape == (3, 20)

    preds = algo(10, ItemList([4]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_fsvd_predict_bad_user():
    algo = funk.FunkSVDScorer(features=20, epochs=20)
    algo.train(simple_ds)

    assert algo.bias is not None
    assert algo.bias.global_bias == approx(simple_df.rating.mean())
    assert algo.item_embeddings.shape == (3, 20)
    assert algo.user_embeddings.shape == (3, 20)

    preds = algo(query=50, items=ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


@mark.slow
def test_fsvd_known_preds(ml_ds: Dataset):
    algo = funk.FunkSVDScorer(features=15, epochs=125, lrate=0.001)
    _log.info("training %s on ml data", algo)
    pipe = predict_pipeline(algo, fallback=False)
    pipe.train(ml_ds)

    dir = Path(__file__).parent
    pred_file = dir / "funksvd-preds.csv"
    _log.info("reading known predictions from %s", pred_file)
    known_preds = pd.read_csv(str(pred_file))
    known = ItemListCollection.from_df(known_preds, UserIDKey)

    preds = batch.predict(pipe, known, n_jobs=1)
    preds = preds.to_df().drop(columns=["prediction"], errors="ignore")

    known_preds.rename(columns={"prediction": "expected"}, inplace=True)
    merged = pd.merge(known_preds, preds)

    merged["error"] = merged.expected - merged.score
    assert not any(merged.score.isna() & merged.expected.notna())
    err = merged.error
    err = err[err.notna()]
    try:
        assert all(err.abs() < 0.01)
    except AssertionError as e:
        bad = merged[merged.error.notna() & (merged.error.abs() >= 0.01)]
        _log.error("erroneous predictions:\n%s", bad)
        raise e
