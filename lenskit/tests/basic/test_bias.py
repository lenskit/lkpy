# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle

import numpy as np
import pandas as pd
import torch

from pytest import approx, mark, raises

from lenskit import util as lku
from lenskit.basic import BiasModel, BiasScorer
from lenskit.data import Dataset, from_interactions_df
from lenskit.data.items import ItemList
from lenskit.operations import predict, recommend
from lenskit.pipeline import Pipeline, PipelineBuilder, topn_pipeline
from lenskit.testing import BasicComponentTests, ScorerTests

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)
simple_ds = from_interactions_df(simple_df)


class TestBias(BasicComponentTests, ScorerTests):
    component = BiasScorer
    needs_jit = False
    configs = [{"damping": 10}, {"damping": {"user": 5, "item": 25}}]
    can_score = "all"


def test_bias_check_arguments():
    # negative damping is not allowed
    with raises(ValueError):
        BiasScorer(damping=-1)

    # negative user damping not allowed
    with raises(ValueError):
        BiasScorer(damping=(-1, 5))

    # negative user damping not allowed
    with raises(ValueError):
        BiasScorer(damping=(5, -1))


def test_bias_full():
    bias = BiasModel.learn(simple_ds)
    assert bias.global_bias == approx(3.5)

    assert bias.items is not None
    assert set(bias.items.ids()) == set([1, 2, 3])
    off = bias.item_biases
    assert off is not None
    exp = pd.Series([0, 1.5, -1.5], index=[1, 2, 3])
    exp = exp.reindex(bias.items.ids())
    df = pd.DataFrame({"computed": off, "expected": exp}).join(
        simple_df.groupby("item")["rating"].mean()
    )
    _log.debug("item stats:\n%s", df)
    assert off == approx(exp.values)

    assert bias.users is not None
    assert set(bias.users.ids()) == set([10, 12, 13])
    off = bias.user_biases
    assert off is not None
    exp = pd.Series([0.25, -0.5, 0], index=[10, 12, 13])
    exp = exp.reindex(bias.users.ids())
    _log.debug("computed user offsets:\n%s", off)
    _log.debug("expected user offsets:\n%s", exp)
    assert off == approx(exp.values)


def test_bias_clone():
    bias = BiasScorer()
    bias.train(simple_ds)

    params = bias.dump_config()
    assert sorted(params.keys()) == ["damping", "entities"]

    a2 = BiasScorer(BiasScorer.validate_config(params))
    assert a2 is not bias
    assert getattr(a2, "model_", None) is None


def test_bias_clone_damping():
    bias = BiasScorer(damping={"user": 10, "item": 5})
    bias.train(simple_ds)

    params = bias.dump_config()
    assert sorted(params.keys()) == ["damping", "entities"]

    a2 = BiasScorer(BiasScorer.validate_config(params))
    assert a2 is not bias
    assert isinstance(a2.config.damping, dict)
    assert a2.config.damping["user"] == 10
    assert a2.config.damping["item"] == 5
    assert getattr(a2, "model_", None) is None


def test_bias_global_only():
    bias = BiasModel.learn(simple_ds, entities=[])
    assert bias.global_bias == approx(3.5)
    assert bias.items is None
    assert bias.item_biases is None
    assert bias.users is None
    assert bias.user_biases is None


def test_bias_no_user():
    bias = BiasModel.learn(simple_ds, entities={"item"})
    assert bias.global_bias == approx(3.5)

    assert bias.item_biases is not None
    assert bias.item_biases == approx(np.array([0, 1.5, -1.5]))

    assert bias.user_biases is None


def test_bias_no_item():
    bias = BiasModel.learn(simple_ds, entities={"user"})
    assert bias.global_bias == approx(3.5)
    assert bias.item_biases is None

    assert bias.user_biases is not None
    assert bias.user_biases == approx(np.array([1.0, -0.5, -1.5]))


def test_bias_global_predict():
    bias = BiasScorer(entities=[])
    bias.train(simple_ds)

    p = bias(10, ItemList(item_ids=[1, 2, 3]))

    assert len(p) == 3
    assert p.scores() == approx(bias.model_.global_bias)


def test_bias_item_predict():
    bias = BiasScorer(entities={"item"})
    bias.train(simple_ds)
    assert bias.model_.item_biases is not None

    p = bias(10, ItemList(item_ids=[1, 2, 3]))

    assert len(p) == 3
    assert p.scores() == approx((bias.model_.item_biases + bias.model_.global_bias))


def test_bias_user_predict():
    bias = BiasScorer(entities={"user"})
    bias.train(simple_ds)
    bm = bias.model_
    p = bias(10, ItemList(item_ids=[1, 2, 3]))

    assert len(p) == 3
    assert p.scores() == approx(bm.global_bias + 1.0)

    p = bias(12, ItemList(item_ids=[1, 3]))

    assert len(p) == 2
    assert p.scores() == approx(bm.global_bias - 0.5)


def test_bias_new_user_predict():
    bias = BiasScorer()
    bias.train(simple_ds)
    bm = bias.model_
    assert bm.item_biases is not None

    items = ItemList(item_ids=[1, 2, 3], rating=[1.5, 2.5, 3.5])
    p = bias(items, ItemList(item_ids=[1, 3]))

    ratings = items.field("rating")
    assert ratings is not None
    offs = ratings - bm.global_bias - bm.item_biases
    umean = offs.mean()
    _log.info("user mean is %f", umean)

    assert len(p) == 2
    assert p.scores() == approx((bm.global_bias + bm.item_biases + umean)[[0, 2]])


def test_bias_predict_unknown_item():
    bias = BiasScorer()
    bias.train(simple_ds)
    bm = bias.model_

    assert bm.items is not None
    assert bm.item_biases is not None

    p = bias(10, ItemList(item_ids=[1, 3, 4]))

    assert len(p) == 3
    locs = bm.items.numbers([1, 3])
    intended = bm.item_biases[locs] + bm.global_bias + 0.25
    ps = p.scores("pandas", index="ids")
    assert ps is not None
    assert ps.loc[[1, 3]].values == approx(intended)
    assert ps.loc[4] == approx(bm.global_bias + 0.25)


def test_bias_predict_unknown_user():
    bias = BiasScorer()
    bias.train(simple_ds)
    bm = bias.model_
    assert bm.items is not None
    assert bm.item_biases is not None

    p = bias(15, ItemList(item_ids=[1, 3]))

    assert len(p) == 2
    locs = bm.items.numbers([1, 3])
    assert p.scores() == approx((bm.item_biases[locs] + bm.global_bias))


def test_bias_train_ml_ratings(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    bias = BiasScorer()
    bias.train(ml_ds)
    bm = bias.model_
    assert bm.items is not None
    assert bm.item_biases is not None

    assert bm.global_bias == approx(ml_ratings.rating.mean())
    imeans_data = ml_ds.item_stats()["mean_rating"]
    imeans_algo = bm.item_biases + bm.global_bias
    ares, data = pd.Series(imeans_algo, index=bm.items.ids()).align(imeans_data)
    assert ares.values == approx(data.values)

    urates = ml_ratings.set_index("user_id").loc[2].set_index("item_id").rating
    umean = (urates - imeans_data[urates.index]).mean()
    p = bias(2, ItemList(item_ids=[10, 11, -1]))
    assert len(p) == 3
    ps = p.scores("pandas", index="ids")
    assert ps is not None
    assert ps.iloc[0] == approx(imeans_data.loc[10] + umean)
    assert ps.iloc[1] == approx(imeans_data.loc[11] + umean)
    assert ps.iloc[2] == approx(ml_ratings.rating.mean() + umean)


def test_bias_item_damp():
    bias = BiasModel.learn(simple_ds, entities={"item"}, damping=5)
    assert bias.global_bias == approx(3.5)

    assert bias.item_biases is not None
    assert bias.item_biases == approx(np.array([0, 0.25, -0.25]))

    assert bias.user_biases is None


def test_bias_user_damp():
    bias = BiasModel.learn(simple_ds, entities={"user"}, damping=5)
    assert bias.global_bias == approx(3.5)
    assert bias.item_biases is None

    assert bias.user_biases is not None
    assert bias.user_biases == approx(np.array([0.2857, -0.08333, -0.25]), abs=1.0e-4)


def test_bias_damped():
    bias = BiasModel.learn(simple_ds, damping=5)
    assert bias.global_bias == approx(3.5)

    assert bias.item_biases is not None
    assert bias.item_biases == approx(np.array([0, 0.25, -0.25]))

    assert bias.user_biases is not None
    assert bias.user_biases == approx(np.array([0.25, -00.08333, -0.20833]), abs=1.0e-4)


def test_bias_separate_damping():
    bias = BiasModel.learn(simple_ds, damping=(5, 10))
    assert bias.global_bias == approx(3.5)

    assert bias.item_biases is not None
    assert bias.item_biases == approx(np.array([0, 0.136364, -0.13636]), abs=1.0e-4)

    assert bias.user_biases is not None
    assert bias.user_biases == approx(np.array([0.266234, -0.08333, -0.22727]), abs=1.0e-4)


def test_bias_save():
    original = BiasScorer(damping=5)
    original.train(simple_ds)
    assert original.model_.global_bias == approx(3.5)

    _log.info("saving baseline model")
    data = pickle.dumps(original)
    _log.info("serialized to %d bytes", len(data))

    copy = pickle.loads(data)

    assert copy.model_.global_bias == original.model_.global_bias
    assert np.all(copy.model_.item_biases == original.model_.item_biases)
    assert np.all(copy.model_.user_biases == original.model_.user_biases)


def test_bias_pipeline(ml_ds: Dataset):
    pipe = PipelineBuilder()
    user = pipe.create_input("user", int)
    items = pipe.create_input("items")

    bias = BiasScorer()
    bias.train(ml_ds)
    out = pipe.add_component("bias", bias, query=user, items=items)

    pipe = pipe.build()
    res = pipe.run(out, user=2, items=ItemList(item_ids=[10, 11, -1]))

    assert len(res) == 3
    assert np.all(res.ids() == [10, 11, -1])

    scores = res.scores()
    assert scores is not None
    assert not np.any(np.isnan(scores[:2]))


def test_bias_topn(ml_ds: Dataset):
    pipe = topn_pipeline(BiasScorer(), predicts_ratings=True, n=10)
    print(pipe.config)
    pipe.train(ml_ds)

    res = predict(pipe, 2, ItemList(item_ids=[10, 11, -1]))
    assert isinstance(res, ItemList)
    assert len(res) == 3
    assert np.all(res.ids() == [10, 11, -1])

    recs = recommend(pipe, 2, n=10)
    assert isinstance(recs, ItemList)
    assert len(recs) == 10


def test_bias_topn_run_length(ml_ds: Dataset):
    pipe = topn_pipeline(BiasScorer(), predicts_ratings=True, n=100)
    print(pipe.config)
    pipe.train(ml_ds)

    res = predict(pipe, 2, items=ItemList(item_ids=[10, 11, -1]))
    assert isinstance(res, ItemList)
    assert len(res) == 3
    assert np.all(res.ids() == [10, 11, -1])

    res = recommend(pipe, 2, n=10)
    assert isinstance(res, ItemList)
    assert len(res) == 10
