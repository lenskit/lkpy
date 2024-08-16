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
from lenskit.algorithms.bias import Bias
from lenskit.data import Dataset, from_interactions_df

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)
simple_ds = from_interactions_df(simple_df)


def test_bias_check_arguments():
    # negative damping is not allowed
    with raises(ValueError):
        Bias(damping=-1)

    # negative user damping not allowed
    with raises(ValueError):
        Bias(damping=(-1, 5))

    # negative user damping not allowed
    with raises(ValueError):
        Bias(damping=(5, -1))


def test_bias_full():
    algo = Bias()
    algo.fit(simple_ds)
    assert algo.mean_ == approx(3.5)

    assert algo.item_offsets_ is not None
    assert algo.item_offsets_.index.name == "item"
    assert set(algo.item_offsets_.index) == set([1, 2, 3])
    exp_item = pd.Series([0, 1.5, -1.5], index=[1, 2, 3])
    off, exp = algo.item_offsets_.align(exp_item)
    df = pd.DataFrame({"computed": off, "expected": exp}).join(
        simple_df.groupby("item")["rating"].mean()
    )
    _log.debug("item stats:\n%s", df)
    assert off.values == approx(exp.values)

    assert algo.user_offsets_ is not None
    assert algo.user_offsets_.index.name == "user"
    assert set(algo.user_offsets_.index) == set([10, 12, 13])
    exp_user = pd.Series([0.25, -0.5, 0], index=[10, 12, 13])
    off, exp = algo.user_offsets_.align(exp_user)
    _log.debug("computed user offsets:\n%s", off)
    _log.debug("expected user offsets:\n%s", exp)
    assert off.values == approx(exp.values)


def test_bias_clone():
    algo = Bias()
    algo.fit(simple_ds)

    params = algo.get_params()
    assert sorted(params.keys()) == ["damping", "items", "users"]

    a2 = lku.clone(algo)
    assert a2 is not algo
    assert getattr(a2, "mean_", None) is None
    assert getattr(a2, "item_offsets_", None) is None
    assert getattr(a2, "user_offsets_", None) is None


def test_bias_clone_damping():
    algo = Bias(damping=(10, 5))
    algo.fit(simple_ds)

    params = algo.get_params()
    assert sorted(params.keys()) == ["damping", "items", "users"]

    a2 = lku.clone(algo)
    assert a2 is not algo
    assert getattr(a2, "item_damping", None) == 5
    assert getattr(a2, "user_damping", None) == 10
    assert getattr(a2, "mean_", None) is None
    assert getattr(a2, "item_offsets_", None) is None
    assert getattr(a2, "user_offsets_", None) is None


def test_bias_global_only():
    algo = Bias(users=False, items=False)
    algo.fit(simple_ds)
    assert algo.mean_ == approx(3.5)
    assert algo.item_offsets_ is None
    assert algo.user_offsets_ is None


def test_bias_no_user():
    algo = Bias(users=False)
    algo.fit(simple_ds)
    assert algo.mean_ == approx(3.5)

    assert algo.item_offsets_ is not None
    assert algo.item_offsets_.index.name == "item"
    assert set(algo.item_offsets_.index) == set([1, 2, 3])
    assert algo.item_offsets_.loc[1:3].values == approx(np.array([0, 1.5, -1.5]))

    assert algo.user_offsets_ is None


def test_bias_no_item():
    algo = Bias(items=False)
    algo.fit(simple_ds)
    assert algo.mean_ == approx(3.5)
    assert algo.item_offsets_ is None

    assert algo.user_offsets_ is not None
    assert algo.user_offsets_.index.name == "user"
    assert set(algo.user_offsets_.index) == set([10, 12, 13])
    assert algo.user_offsets_.loc[[10, 12, 13]].values == approx(np.array([1.0, -0.5, -1.5]))


def test_bias_index_props():
    algo = Bias()
    algo.fit(simple_ds)
    assert all(np.sort(algo.user_index) == np.unique(simple_df["user"]))
    assert all(np.sort(algo.item_index) == np.unique(simple_df["item"]))


def test_bias_global_predict():
    algo = Bias(items=False, users=False)
    algo.fit(simple_ds)
    p = algo.predict_for_user(10, [1, 2, 3])

    assert len(p) == 3
    assert (p == algo.mean_).all()
    assert p.values == approx(algo.mean_)


def test_bias_item_predict():
    algo = Bias(users=False)
    algo.fit(simple_ds)
    assert algo.item_offsets_ is not None

    p = algo.predict_for_user(10, [1, 2, 3])

    assert len(p) == 3
    assert p.values == approx((algo.item_offsets_ + algo.mean_).values)


def test_bias_user_predict():
    algo = Bias(items=False)
    algo.fit(simple_ds)
    p = algo.predict_for_user(10, [1, 2, 3])

    assert len(p) == 3
    assert p.values == approx(algo.mean_ + 1.0)

    p = algo.predict_for_user(12, [1, 3])

    assert len(p) == 2
    assert p.values == approx(algo.mean_ - 0.5)


def test_bias_new_user_predict():
    algo = Bias()
    algo.fit(simple_ds)
    assert algo.item_offsets_ is not None

    ratings = pd.DataFrame({"item": [1, 2, 3], "rating": [1.5, 2.5, 3.5]})
    ratings = ratings.set_index("item").rating
    p = algo.predict_for_user(None, [1, 3], ratings=ratings)

    offs = ratings - algo.mean_ - algo.item_offsets_
    umean = offs.mean()
    _log.info("user mean is %f", umean)

    assert len(p) == 2
    assert p.values == approx((algo.mean_ + algo.item_offsets_ + umean).loc[[1, 3]].values)


def test_bias_predict_unknown_item():
    algo = Bias()
    algo.fit(simple_ds)
    assert algo.item_offsets_ is not None

    p = algo.predict_for_user(10, [1, 3, 4])

    assert len(p) == 3
    intended = algo.item_offsets_.loc[[1, 3]] + algo.mean_ + 0.25
    assert p.loc[[1, 3]].values == approx(intended.values)
    assert p.loc[4] == approx(algo.mean_ + 0.25)


def test_bias_predict_unknown_user():
    algo = Bias()
    algo.fit(simple_ds)
    assert algo.item_offsets_ is not None

    p = algo.predict_for_user(15, [1, 3])

    assert len(p) == 2
    assert p.values == approx((algo.item_offsets_.loc[[1, 3]] + algo.mean_).values)


def test_bias_train_ml_ratings(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    algo = Bias()
    algo.fit(ml_ds)
    assert algo.item_offsets_ is not None

    assert algo.mean_ == approx(ml_ratings.rating.mean())
    imeans_data = ml_ds.item_stats()["mean_rating"]
    imeans_algo = algo.item_offsets_ + algo.mean_
    ares, data = imeans_algo.align(imeans_data)
    assert ares.values == approx(data.values)

    urates = ml_ratings.set_index("user").loc[2].set_index("item").rating
    umean = (urates - imeans_data[urates.index]).mean()
    p = algo.predict_for_user(2, [10, 11, -1])
    assert len(p) == 3
    assert p.iloc[0] == approx(imeans_data.loc[10] + umean)
    assert p.iloc[1] == approx(imeans_data.loc[11] + umean)
    assert p.iloc[2] == approx(ml_ratings.rating.mean() + umean)


def test_bias_transform(ml_ds: Dataset):
    algo = Bias()

    normed = algo.fit_transform(ml_ds)

    ratings = ml_ds.interaction_log("pandas", original_ids=True)
    assert all(normed["user"] == ratings["user_id"])
    assert all(normed["item"] == ratings["item_id"])
    denorm = algo.inverse_transform(normed)
    assert denorm["rating"].values == approx(ratings["rating"], 1.0e-6)

    assert algo.item_offsets_ is not None
    assert algo.user_offsets_ is not None
    n2 = ratings.join(algo.item_offsets_, on="item_id")
    n2 = n2.join(algo.user_offsets_, on="user_id")
    nr = n2.rating - algo.mean_ - n2.i_off - n2.u_off
    assert normed["rating"].values == approx(nr.values)


def test_bias_transform_tensor(ml_ds):
    algo = Bias()

    algo.fit(ml_ds)
    assert algo.item_offsets_ is not None
    assert algo.user_offsets_ is not None

    mat = ml_ds.interaction_matrix("torch", layout="coo")
    normed = algo.transform(mat)
    assert normed.is_sparse
    assert normed.shape == mat.shape

    recon = normed.clone()
    recon.values().add_(algo.mean_)
    recon.values().add_(torch.from_numpy(algo.item_offsets_.values)[recon.indices()[1, :]])
    recon.values().add_(torch.from_numpy(algo.user_offsets_.values)[recon.indices()[0, :]])

    assert recon.values().numpy() == approx(mat.values().numpy())


def test_bias_transform_indexes(ml_ds: Dataset):
    algo = Bias()

    normed = algo.fit_transform(ml_ds, indexes=True)
    assert algo.item_offsets_ is not None
    assert algo.user_offsets_ is not None

    ratings = ml_ds.interaction_log("pandas", original_ids=True)

    assert all(normed["user"] == ratings["user_id"])
    assert all(normed["item"] == ratings["item_id"])
    assert all(normed["uidx"] == ml_ds.users.numbers(ratings["user_id"]))
    assert all(normed["iidx"] == ml_ds.items.numbers(ratings["item_id"]))
    denorm = algo.inverse_transform(normed)
    assert denorm["rating"].values == approx(ratings["rating"].values, 1.0e-6)


@mark.parametrize(["users", "items"], [(True, False), (False, True), (False, False)])
def test_bias_transform_disable(ml_ds: Dataset, users: bool, items: bool):
    algo = Bias(users=users, items=items)

    normed = algo.fit_transform(ml_ds)

    ratings = ml_ds.interaction_log("pandas", original_ids=True)
    assert all(normed["user"] == ratings["user_id"])
    assert all(normed["item"] == ratings["item_id"])
    denorm = algo.inverse_transform(normed)
    assert denorm["rating"].values == approx(ratings["rating"], 1.0e-6)

    n2 = ratings
    nr = n2.rating - algo.mean_
    if items:
        assert algo.item_offsets_ is not None
        n2 = n2.join(algo.item_offsets_, on="item_id")
        nr = nr - n2.i_off
    else:
        assert algo.item_offsets_ is None
    if users:
        assert algo.user_offsets_ is not None
        n2 = n2.join(algo.user_offsets_, on="user_id")
        nr = nr - n2.u_off
    else:
        assert algo.user_offsets_ is None
    assert normed["rating"].values == approx(nr.values)


def test_bias_item_damp():
    algo = Bias(users=False, damping=5)
    algo.fit(simple_ds)
    assert algo.mean_ == approx(3.5)

    assert algo.item_offsets_ is not None
    assert algo.item_offsets_.index.name == "item"
    assert set(algo.item_offsets_.index) == set([1, 2, 3])
    assert algo.item_offsets_.loc[1:3].values == approx(np.array([0, 0.25, -0.25]))

    assert algo.user_offsets_ is None


def test_bias_user_damp():
    algo = Bias(items=False, damping=5)
    algo.fit(simple_ds)
    assert algo.mean_ == approx(3.5)
    assert algo.item_offsets_ is None

    assert algo.user_offsets_ is not None
    assert algo.user_offsets_.index.name == "user"
    assert set(algo.user_offsets_.index) == set([10, 12, 13])
    assert algo.user_offsets_.loc[[10, 12, 13]].values == approx(
        np.array([0.2857, -0.08333, -0.25]), abs=1.0e-4
    )


def test_bias_damped():
    algo = Bias(damping=5)
    algo.fit(simple_ds)
    assert algo.mean_ == approx(3.5)

    assert algo.item_offsets_ is not None
    assert algo.item_offsets_.index.name == "item"
    assert set(algo.item_offsets_.index) == set([1, 2, 3])
    assert algo.item_offsets_.loc[1:3].values == approx(np.array([0, 0.25, -0.25]))

    assert algo.user_offsets_ is not None
    assert algo.user_offsets_.index.name == "user"
    assert set(algo.user_offsets_.index) == set([10, 12, 13])
    assert algo.user_offsets_.loc[[10, 12, 13]].values == approx(
        np.array([0.25, -00.08333, -0.20833]), abs=1.0e-4
    )


def test_bias_separate_damping():
    algo = Bias(damping=(5, 10))
    algo.fit(simple_ds)
    assert algo.mean_ == approx(3.5)

    assert algo.item_offsets_ is not None
    assert algo.item_offsets_.index.name == "item"
    assert set(algo.item_offsets_.index) == set([1, 2, 3])
    assert algo.item_offsets_.loc[1:3].values == approx(
        np.array([0, 0.136364, -0.13636]), abs=1.0e-4
    )

    assert algo.user_offsets_ is not None
    assert algo.user_offsets_.index.name == "user"
    assert set(algo.user_offsets_.index) == set([10, 12, 13])
    assert algo.user_offsets_.loc[[10, 12, 13]].values == approx(
        np.array([0.266234, -0.08333, -0.22727]), abs=1.0e-4
    )


def test_transform_user_with_user_bias():
    algo = Bias()
    algo.fit(simple_ds)

    new_ratings = pd.Series([4.0, 5.0], index=[1, 2])  # items as index and ratings as values

    ratings_with_bias, user_bias = algo.transform_user(new_ratings)  # user: 13
    result = algo.inverse_transform_user(13, ratings_with_bias, user_bias)

    assert new_ratings[1] == result[1]
    assert new_ratings[2] == result[2]


def test_transform_user_without_user_bias():
    user = 12
    algo = Bias()
    algo.fit(simple_ds)
    assert algo.item_offsets_ is not None
    assert algo.user_offsets_ is not None

    new_ratings = pd.Series([-0.5, 1.5], index=[2, 3])  # items as index and ratings as values

    v = algo.inverse_transform_user(user, new_ratings)

    assert (
        v[2]
        == new_ratings[2] + algo.user_offsets_.loc[user] + algo.item_offsets_.loc[2] + algo.mean_
    )
    assert (
        v[3]
        == new_ratings[3] + algo.user_offsets_.loc[user] + algo.item_offsets_.loc[3] + algo.mean_
    )


def test_bias_save():
    original = Bias(damping=5)
    original.fit(simple_ds)
    assert original.mean_ == approx(3.5)

    _log.info("saving baseline model")
    mod = pickle.dumps(original)
    _log.info("serialized to %d bytes", len(mod))

    algo = pickle.loads(mod)

    assert algo.mean_ == original.mean_

    assert algo.item_offsets_ is not None
    assert algo.item_offsets_.index.name == "item"
    assert set(algo.item_offsets_.index) == set([1, 2, 3])
    assert algo.item_offsets_.loc[1:3].values == approx(np.array([0, 0.25, -0.25]))

    assert algo.user_offsets_ is not None
    assert algo.user_offsets_.index.name == "user"
    assert set(algo.user_offsets_.index) == set([10, 12, 13])
    assert algo.user_offsets_.loc[[10, 12, 13]].values == approx(
        np.array([0.25, -00.08333, -0.20833]), abs=1.0e-4
    )
