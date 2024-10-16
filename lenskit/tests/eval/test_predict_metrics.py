# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from pytest import approx, mark, raises

import lenskit.metrics.predict as pm
import lenskit.util.test as lktu
from lenskit.data import ItemList, from_interactions_df


def test_check_missing_empty():
    pm._check_missing(pd.Series([], dtype="float64"), "error")
    # should pass
    assert True


def test_check_missing_has_values():
    pm._check_missing(pd.Series([1, 3, 2]), "error")
    # should pass
    assert True


def test_check_missing_nan_raises():
    with raises(ValueError):
        pm._check_missing(pd.Series([1, np.nan, 3]), "error")


def test_check_missing_raises():
    data = pd.Series([1, 7, 3], ["a", "b", "d"])
    ref = pd.Series([3, 2, 4], ["b", "c", "d"])
    ref, data = ref.align(data, join="left")
    with raises(ValueError):
        pm._check_missing(data, "error")


def test_check_joined_ok():
    data = pd.Series([1, 7, 3], ["a", "b", "d"])
    ref = pd.Series([3, 2, 4], ["b", "c", "d"])
    ref, data = ref.align(data, join="inner")
    pm._check_missing(ref, "error")
    # should get here
    assert True


def test_check_missing_ignore():
    data = pd.Series([1, 7, 3], ["a", "b", "d"])
    ref = pd.Series([3, 2, 4], ["b", "c", "d"])
    ref, data = ref.align(data, join="left")
    pm._check_missing(data, "ignore")
    # should get here
    assert True


def test_rmse_one():
    rmse = pm.rmse(ItemList(["a"], scores=[1]), ItemList(["a"], rating=[1]))
    assert isinstance(rmse, float)
    assert rmse == approx(0)

    rmse = pm.rmse(ItemList(["a"], scores=[1]), ItemList(["a"], rating=[2]))
    assert rmse == approx(1)

    rmse = pm.rmse(ItemList(["a"], scores=[1]), ItemList(["a"], rating=[0.5]))
    assert rmse == approx(0.5)


def test_rmse_two():
    rmse = pm.rmse(ItemList(["a", "b"], scores=[1, 2]), ItemList(["a", "b"], rating=[1, 2]))
    assert isinstance(rmse, float)
    assert rmse == approx(0)

    rmse = pm.rmse(ItemList(["a", "b"], scores=[1, 1]), ItemList(["a", "b"], rating=[2, 2]))
    assert rmse == approx(1)

    rmse = pm.rmse(ItemList(["a", "b"], scores=[1, 3]), ItemList(["a", "b"], rating=[3, 1]))
    assert rmse == approx(2)


def test_rmse_series_subset_items():
    rmse = pm.rmse(
        ItemList(scores=[1, 3], item_ids=["a", "c"]),
        ItemList(rating=[3, 4, 1], item_ids=["a", "b", "c"]),
    )
    assert rmse == approx(2)


def test_rmse_series_missing_value_error():
    with raises(ValueError):
        pm.rmse(
            ItemList(scores=[1, 3], item_ids=["a", "d"]),
            ItemList(rating=[3, 4, 1], item_ids=["a", "b", "c"]),
        )


def test_rmse_series_missing_value_ignore():
    rmse = pm.rmse(
        ItemList(scores=[1, 3], item_ids=["a", "d"]),
        ItemList(rating=[3, 4, 1], item_ids=["a", "b", "c"]),
        missing="ignore",
    )
    assert rmse == approx(2)


def test_mae_two():
    mae = pm.mae([1, 2], [1, 2])
    assert isinstance(mae, float)
    assert mae == approx(0)

    mae = pm.mae([1, 1], [2, 2])
    assert mae == approx(1)

    mae = pm.mae([1, 3], [3, 1])
    assert mae == approx(2)

    mae = pm.mae([1, 3], [3, 2])
    assert mae == approx(1.5)


def test_mae_array_two():
    mae = pm.mae(np.array([1, 2]), np.array([1, 2]))
    assert isinstance(mae, float)
    assert mae == approx(0)

    mae = pm.mae(np.array([1, 1]), np.array([2, 2]))
    assert mae == approx(1)

    mae = pm.mae(np.array([1, 3]), np.array([3, 1]))
    assert mae == approx(2)


def test_mae_series_two():
    mae = pm.mae(pd.Series([1, 2]), pd.Series([1, 2]))
    assert isinstance(mae, float)
    assert mae == approx(0)

    mae = pm.mae(pd.Series([1, 1]), pd.Series([2, 2]))
    assert mae == approx(1)

    mae = pm.mae(pd.Series([1, 3]), pd.Series([3, 1]))
    assert mae == approx(2)


@mark.slow
@mark.eval
def test_batch_rmse(ml_100k):
    import lenskit.algorithms.bias as bs
    import lenskit.batch as batch
    import lenskit.crossfold as xf

    algo = bs.Bias(damping=5)

    def eval(train, test):
        algo.fit(from_interactions_df(train))
        preds = batch.predict(algo, test)
        return preds.set_index(["user", "item"])

    results = pd.concat(
        (eval(train, test) for (train, test) in xf.partition_users(ml_100k, 5, xf.SampleN(5)))
    )

    user_rmse = results.groupby("user").apply(lambda df: pm.rmse(df.prediction, df.rating))

    # we should have all users
    users = ml_100k.user.unique()
    assert len(user_rmse) == len(users)
    missing = np.setdiff1d(users, user_rmse.index)
    assert len(missing) == 0

    # we should not have any missing values
    assert all(user_rmse.notna())

    # we should have a reasonable mean
    assert user_rmse.mean() == approx(0.93, abs=0.05)
