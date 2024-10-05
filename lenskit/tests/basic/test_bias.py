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
from lenskit.basic import BiasScorer
from lenskit.data import Dataset, from_interactions_df
from lenskit.data.items import ItemList
from lenskit.pipeline import Pipeline
from lenskit.pipeline.common import topn_pipeline

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)
simple_ds = from_interactions_df(simple_df)


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
    bias = BiasScorer()
    bias.train(simple_ds)
    assert bias.mean_ == approx(3.5)

    assert bias.items_ is not None
    assert set(bias.items_.ids()) == set([1, 2, 3])
    off = bias.item_offsets_
    assert off is not None
    exp = pd.Series([0, 1.5, -1.5], index=[1, 2, 3])
    exp = exp.reindex(bias.items_.ids())
    df = pd.DataFrame({"computed": off, "expected": exp}).join(
        simple_df.groupby("item")["rating"].mean()
    )
    _log.debug("item stats:\n%s", df)
    assert off == approx(exp.values)

    assert bias.users_ is not None
    assert set(bias.users_.ids()) == set([10, 12, 13])
    off = bias.user_offsets_
    assert off is not None
    exp = pd.Series([0.25, -0.5, 0], index=[10, 12, 13])
    exp = exp.reindex(bias.users_.ids())
    _log.debug("computed user offsets:\n%s", off)
    _log.debug("expected user offsets:\n%s", exp)
    assert off == approx(exp.values)


def test_bias_clone():
    bias = BiasScorer()
    bias.train(simple_ds)

    params = bias.get_config()
    assert sorted(params.keys()) == ["damping", "items", "users"]

    a2 = BiasScorer.from_config(params)
    assert a2 is not bias
    assert getattr(a2, "mean_", None) is None
    assert getattr(a2, "item_offsets_", None) is None
    assert getattr(a2, "user_offsets_", None) is None


def test_bias_clone_damping():
    bias = BiasScorer(damping=(10, 5))
    bias.train(simple_ds)

    params = bias.get_config()
    assert sorted(params.keys()) == ["damping", "items", "users"]

    a2 = BiasScorer.from_config(params)
    assert a2 is not bias
    assert a2.damping.user == 10
    assert a2.damping.item == 5
    assert getattr(a2, "mean_", None) is None
    assert getattr(a2, "item_offsets_", None) is None
    assert getattr(a2, "user_offsets_", None) is None


def test_bias_global_only():
    bias = BiasScorer(users=False, items=False)
    bias.train(simple_ds)
    assert bias.mean_ == approx(3.5)
    assert bias.items_ is None
    assert bias.item_offsets_ is None
    assert bias.users_ is None
    assert bias.user_offsets_ is None


def test_bias_no_user():
    bias = BiasScorer(users=False)
    bias.train(simple_ds)
    assert bias.mean_ == approx(3.5)

    assert bias.item_offsets_ is not None
    assert bias.item_offsets_ == approx(np.array([0, 1.5, -1.5]))

    assert bias.user_offsets_ is None


def test_bias_no_item():
    bias = BiasScorer(items=False)
    bias.train(simple_ds)
    assert bias.mean_ == approx(3.5)
    assert bias.item_offsets_ is None

    assert bias.user_offsets_ is not None
    assert bias.user_offsets_ == approx(np.array([1.0, -0.5, -1.5]))


def test_bias_global_predict():
    bias = BiasScorer(items=False, users=False)
    bias.train(simple_ds)
    p = bias(10, ItemList(item_ids=[1, 2, 3]))

    assert len(p) == 3
    assert p.scores() == approx(bias.mean_)


def test_bias_item_predict():
    bias = BiasScorer(users=False)
    bias.train(simple_ds)
    assert bias.item_offsets_ is not None

    p = bias(10, ItemList(item_ids=[1, 2, 3]))

    assert len(p) == 3
    assert p.scores() == approx((bias.item_offsets_ + bias.mean_))


def test_bias_user_predict():
    bias = BiasScorer(items=False)
    bias.train(simple_ds)
    p = bias(10, ItemList(item_ids=[1, 2, 3]))

    assert len(p) == 3
    assert p.scores() == approx(bias.mean_ + 1.0)

    p = bias(12, ItemList(item_ids=[1, 3]))

    assert len(p) == 2
    assert p.scores() == approx(bias.mean_ - 0.5)


def test_bias_new_user_predict():
    bias = BiasScorer()
    bias.train(simple_ds)
    assert bias.item_offsets_ is not None

    items = ItemList(item_ids=[1, 2, 3], rating=[1.5, 2.5, 3.5])
    p = bias(items, ItemList(item_ids=[1, 3]))

    ratings = items.field("rating")
    assert ratings is not None
    offs = ratings - bias.mean_ - bias.item_offsets_
    umean = offs.mean()
    _log.info("user mean is %f", umean)

    assert len(p) == 2
    assert p.scores() == approx((bias.mean_ + bias.item_offsets_ + umean)[[0, 2]])


def test_bias_predict_unknown_item():
    bias = BiasScorer()
    bias.train(simple_ds)
    assert bias.items_ is not None
    assert bias.item_offsets_ is not None

    p = bias(10, ItemList(item_ids=[1, 3, 4]))

    assert len(p) == 3
    locs = bias.items_.numbers([1, 3])
    intended = bias.item_offsets_[locs] + bias.mean_ + 0.25
    ps = p.scores("pandas", index="ids")
    assert ps is not None
    assert ps.loc[[1, 3]].values == approx(intended)
    assert ps.loc[4] == approx(bias.mean_ + 0.25)


def test_bias_predict_unknown_user():
    bias = BiasScorer()
    bias.train(simple_ds)
    assert bias.items_ is not None
    assert bias.item_offsets_ is not None

    p = bias(15, ItemList(item_ids=[1, 3]))

    assert len(p) == 2
    locs = bias.items_.numbers([1, 3])
    assert p.scores() == approx((bias.item_offsets_[locs] + bias.mean_))


def test_bias_train_ml_ratings(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    bias = BiasScorer()
    bias.train(ml_ds)
    assert bias.items_ is not None
    assert bias.item_offsets_ is not None

    assert bias.mean_ == approx(ml_ratings.rating.mean())
    imeans_data = ml_ds.item_stats()["mean_rating"]
    imeans_algo = bias.item_offsets_ + bias.mean_
    ares, data = pd.Series(imeans_algo, index=bias.items_.ids()).align(imeans_data)
    assert ares.values == approx(data.values)

    urates = ml_ratings.set_index("user").loc[2].set_index("item").rating
    umean = (urates - imeans_data[urates.index]).mean()
    p = bias(2, ItemList(item_ids=[10, 11, -1]))
    assert len(p) == 3
    ps = p.scores("pandas", index="ids")
    assert ps is not None
    assert ps.iloc[0] == approx(imeans_data.loc[10] + umean)
    assert ps.iloc[1] == approx(imeans_data.loc[11] + umean)
    assert ps.iloc[2] == approx(ml_ratings.rating.mean() + umean)


def test_bias_item_damp():
    bias = BiasScorer(users=False, damping=5)
    bias.train(simple_ds)
    assert bias.mean_ == approx(3.5)

    assert bias.item_offsets_ is not None
    assert bias.item_offsets_ == approx(np.array([0, 0.25, -0.25]))

    assert bias.user_offsets_ is None


def test_bias_user_damp():
    bias = BiasScorer(items=False, damping=5)
    bias.train(simple_ds)
    assert bias.mean_ == approx(3.5)
    assert bias.item_offsets_ is None

    assert bias.user_offsets_ is not None
    assert bias.user_offsets_ == approx(np.array([0.2857, -0.08333, -0.25]), abs=1.0e-4)


def test_bias_damped():
    bias = BiasScorer(damping=5)
    bias.train(simple_ds)
    assert bias.mean_ == approx(3.5)

    assert bias.item_offsets_ is not None
    assert bias.item_offsets_ == approx(np.array([0, 0.25, -0.25]))

    assert bias.user_offsets_ is not None
    assert bias.user_offsets_ == approx(np.array([0.25, -00.08333, -0.20833]), abs=1.0e-4)


def test_bias_separate_damping():
    bias = BiasScorer(damping=(5, 10))
    bias.train(simple_ds)
    assert bias.mean_ == approx(3.5)

    assert bias.item_offsets_ is not None
    assert bias.item_offsets_ == approx(np.array([0, 0.136364, -0.13636]), abs=1.0e-4)

    assert bias.user_offsets_ is not None
    assert bias.user_offsets_ == approx(np.array([0.266234, -0.08333, -0.22727]), abs=1.0e-4)


def test_bias_save():
    original = BiasScorer(damping=5)
    original.train(simple_ds)
    assert original.mean_ == approx(3.5)

    _log.info("saving baseline model")
    data = pickle.dumps(original)
    _log.info("serialized to %d bytes", len(data))

    copy = pickle.loads(data)

    assert copy.mean_ == original.mean_

    assert copy.item_offsets_ is not None
    assert copy.item_offsets_ == approx(np.array([0, 0.25, -0.25]))

    assert copy.user_offsets_ is not None
    assert copy.user_offsets_ == approx(np.array([0.25, -00.08333, -0.20833]), abs=1.0e-4)


def test_bias_pipeline(ml_ds: Dataset):
    pipe = Pipeline()
    user = pipe.create_input("user", int)
    items = pipe.create_input("items")

    bias = BiasScorer()
    bias.train(ml_ds)
    out = pipe.add_component("bias", bias, query=user, items=items)

    res = pipe.run(out, user=2, items=ItemList(item_ids=[10, 11, -1]))

    assert len(res) == 3
    assert np.all(res.ids() == [10, 11, -1])

    scores = res.scores()
    assert scores is not None
    assert not np.any(np.isnan(scores[:2]))


def test_bias_topn(ml_ds: Dataset):
    pipe = topn_pipeline(BiasScorer(), predicts_ratings=True)
    print(pipe.get_config())
    pipe.train(ml_ds)

    res = pipe.run("rating-predictor", user=2, items=ItemList(item_ids=[10, 11, -1]))
    assert isinstance(res, ItemList)
    assert len(res) == 3
    assert np.all(res.ids() == [10, 11, -1])

    res = pipe.run("ranker", user=2, n=10)
    assert isinstance(res, ItemList)
    assert len(res) == 10
