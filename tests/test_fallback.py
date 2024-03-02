# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.algorithms import basic
from lenskit.algorithms.bias import Bias
from lenskit import util as lku

import pandas as pd
import numpy as np
import binpickle

import lenskit.util.test as lktu
from pytest import approx

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)


def test_fallback_train_one():
    algo = basic.Fallback(Bias())
    algo.fit(lktu.ml_test.ratings)
    assert len(algo.algorithms) == 1
    assert isinstance(algo.algorithms[0], Bias)
    assert algo.algorithms[0].mean_ == approx(lktu.ml_test.ratings.rating.mean())


def test_fallback_train_one_pred_impossible():
    algo = basic.Fallback(basic.Memorized(simple_df))
    algo.fit(lktu.ml_test.ratings)

    preds = algo.predict_for_user(10, [1, 2])
    assert set(preds.index) == set([1, 2])
    assert all(preds == pd.Series({1: 4.0, 2: 5.0}))

    preds = algo.predict_for_user(12, [1, 3])
    assert set(preds.index) == set([1, 3])
    assert preds.loc[1] == 3.0
    assert np.isnan(preds.loc[3])


def test_fallback_list():
    algo = basic.Fallback([basic.Memorized(simple_df), Bias()])
    algo.fit(lktu.ml_test.ratings)
    assert len(algo.algorithms) == 2

    params = algo.get_params()
    assert list(params.keys()) == ["algorithms"]
    assert len(params["algorithms"]) == 2
    assert isinstance(params["algorithms"][0], basic.Memorized)
    assert isinstance(params["algorithms"][1], Bias)


def test_fallback_string():
    algo = basic.Fallback([basic.Memorized(simple_df), Bias()])
    assert "Fallback" in str(algo)


def test_fallback_clone():
    algo = basic.Fallback([basic.Memorized(simple_df), Bias()])
    algo.fit(lktu.ml_test.ratings)
    assert len(algo.algorithms) == 2

    clone = lku.clone(algo)
    assert clone is not algo
    for a1, a2 in zip(algo.algorithms, clone.algorithms):
        assert a1 is not a2
        assert type(a2) == type(a1)


def test_fallback_predict():
    algo = basic.Fallback(basic.Memorized(simple_df), Bias())
    algo.fit(lktu.ml_test.ratings)
    assert len(algo.algorithms) == 2

    bias = algo.algorithms[1]
    assert isinstance(bias, Bias)
    assert bias.mean_ == approx(lktu.ml_test.ratings.rating.mean())

    def exp_val(user, item):
        v = bias.mean_
        if user is not None:
            v += bias.user_offsets_.loc[user]
        if item is not None:
            v += bias.item_offsets_.loc[item]
        return v

    # first user + item
    preds = algo.predict_for_user(10, [1])
    assert preds.loc[1] == 4.0
    # second user + first item
    preds = algo.predict_for_user(15, [1])
    assert preds.loc[1] == approx(exp_val(15, 1))

    # second item + user item
    preds = algo.predict_for_user(12, [2])
    assert preds.loc[2] == approx(exp_val(12, 2))

    # blended
    preds = algo.predict_for_user(10, [1, 5])
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(exp_val(10, 5))

    # blended unknown
    preds = algo.predict_for_user(10, [5, 1, -23081])
    assert len(preds) == 3
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(exp_val(10, 5))
    assert preds.loc[-23081] == approx(exp_val(10, None))


def test_fallback_save_load(tmp_path):
    original = basic.Fallback(basic.Memorized(simple_df), Bias())
    original.fit(lktu.ml_test.ratings)

    fn = tmp_path / "fb.mod"

    binpickle.dump(original, fn)

    algo = binpickle.load(fn)

    bias = algo.algorithms[1]
    assert bias.mean_ == approx(lktu.ml_test.ratings.rating.mean())

    def exp_val(user, item):
        v = bias.mean_
        if user is not None:
            v += bias.user_offsets_.loc[user]
        if item is not None:
            v += bias.item_offsets_.loc[item]
        return v

    # first user + item
    preds = algo.predict_for_user(10, [1])
    assert preds.loc[1] == 4.0
    # second user + first item
    preds = algo.predict_for_user(15, [1])
    assert preds.loc[1] == approx(exp_val(15, 1))

    # second item + user item
    preds = algo.predict_for_user(12, [2])
    assert preds.loc[2] == approx(exp_val(12, 2))

    # blended
    preds = algo.predict_for_user(10, [1, 5])
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(exp_val(10, 5))

    # blended unknown
    preds = algo.predict_for_user(10, [5, 1, -23081])
    assert len(preds) == 3
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(exp_val(10, 5))
    assert preds.loc[-23081] == approx(exp_val(10, None))
