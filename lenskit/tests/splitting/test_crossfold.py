# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import functools as ft
import itertools as it
import math

import numpy as np
import pandas as pd

import pytest

import lenskit.crossfold as xf


def test_partition_rows(ml_ratings: pd.DataFrame):
    splits = xf.partition_rows(ml_ratings, 5)
    splits = list(splits)
    assert len(splits) == 5

    for s in splits:
        assert len(s.test) + len(s.train) == len(ml_ratings)
        assert all(s.test.index.union(s.train.index) == ml_ratings.index)
        test_idx = s.test.set_index(["user", "item"]).index
        train_idx = s.train.set_index(["user", "item"]).index
        assert len(test_idx.intersection(train_idx)) == 0

    # we should partition!
    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue

        i1 = s1.test.set_index(["user", "item"]).index
        i2 = s2.test.set_index(["user", "item"]).index
        inter = i1.intersection(i2)
        assert len(inter) == 0

    union = ft.reduce(lambda i1, i2: i1.union(i2), (s.test.index for s in splits))
    assert len(union.unique()) == len(ml_ratings)


def test_sample_rows(ml_ratings: pd.DataFrame):
    splits = xf.sample_rows(ml_ratings, partitions=5, size=1000)
    splits = list(splits)
    assert len(splits) == 5

    for s in splits:
        assert len(s.test) == 1000
        assert len(s.test) + len(s.train) == len(ml_ratings)
        test_idx = s.test.set_index(["user", "item"]).index
        train_idx = s.train.set_index(["user", "item"]).index
        assert len(test_idx.intersection(train_idx)) == 0

    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue

        i1 = s1.test.set_index(["user", "item"]).index
        i2 = s2.test.set_index(["user", "item"]).index
        inter = i1.intersection(i2)
        assert len(inter) == 0


def test_sample_rows_more_smaller_parts(ml_ratings: pd.DataFrame):
    splits = xf.sample_rows(ml_ratings, partitions=10, size=500)
    splits = list(splits)
    assert len(splits) == 10

    for s in splits:
        assert len(s.test) == 500
        assert len(s.test) + len(s.train) == len(ml_ratings)
        test_idx = s.test.set_index(["user", "item"]).index
        train_idx = s.train.set_index(["user", "item"]).index
        assert len(test_idx.intersection(train_idx)) == 0

    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue

        i1 = s1.test.set_index(["user", "item"]).index
        i2 = s2.test.set_index(["user", "item"]).index
        inter = i1.intersection(i2)
        assert len(inter) == 0


def test_sample_non_disjoint(ml_ratings: pd.DataFrame):
    splits = xf.sample_rows(ml_ratings, partitions=10, size=1000, disjoint=False)
    splits = list(splits)
    assert len(splits) == 10

    for s in splits:
        assert len(s.test) == 1000
        assert len(s.test) + len(s.train) == len(ml_ratings)
        test_idx = s.test.set_index(["user", "item"]).index
        train_idx = s.train.set_index(["user", "item"]).index
        assert len(test_idx.intersection(train_idx)) == 0

    # There are enough splits & items we should pick at least one duplicate
    ipairs = (
        (s1.test.set_index(["user", "item"]).index, s2.test.set_index(["user", "item"]).index)
        for (s1, s2) in it.product(splits, splits)
    )
    isizes = [len(i1.intersection(i2)) for (i1, i2) in ipairs]
    assert any(n > 0 for n in isizes)


@pytest.mark.slow
def test_sample_oversize(ml_ratings: pd.DataFrame):
    splits = xf.sample_rows(ml_ratings, 50, 10000)
    splits = list(splits)
    assert len(splits) == 50

    for s in splits:
        assert len(s.test) + len(s.train) == len(ml_ratings)
        assert all(s.test.index.union(s.train.index) == ml_ratings.index)
        test_idx = s.test.set_index(["user", "item"]).index
        train_idx = s.train.set_index(["user", "item"]).index
        assert len(test_idx.intersection(train_idx)) == 0


def test_sample_n(ml_ratings: pd.DataFrame):
    users = np.random.choice(ml_ratings.user.unique(), 5, replace=False)

    s5 = xf.SampleN(5)
    for u in users:
        udf = ml_ratings[ml_ratings.user == u]
        tst = s5(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) == 5
        assert len(tst) + len(trn) == len(udf)

    s10 = xf.SampleN(10)
    for u in users:
        udf = ml_ratings[ml_ratings.user == u]
        tst = s10(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) == 10
        assert len(tst) + len(trn) == len(udf)


def test_sample_frac(ml_ratings: pd.DataFrame):
    users = np.random.choice(ml_ratings.user.unique(), 5, replace=False)

    samp = xf.SampleFrac(0.2)
    for u in users:
        udf = ml_ratings[ml_ratings.user == u]
        tst = samp(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) + len(trn) == len(udf)
        assert len(tst) >= math.floor(len(udf) * 0.2)
        assert len(tst) <= math.ceil(len(udf) * 0.2)

    samp = xf.SampleFrac(0.5)
    for u in users:
        udf = ml_ratings[ml_ratings.user == u]
        tst = samp(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) + len(trn) == len(udf)
        assert len(tst) >= math.floor(len(udf) * 0.5)
        assert len(tst) <= math.ceil(len(udf) * 0.5)


def test_last_n(ml_ratings: pd.DataFrame):
    users = np.random.choice(ml_ratings.user.unique(), 5, replace=False)

    samp = xf.LastN(5)
    for u in users:
        udf = ml_ratings[ml_ratings.user == u]
        tst = samp(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) == 5
        assert len(tst) + len(trn) == len(udf)
        assert tst.timestamp.min() >= trn.timestamp.max()

    samp = xf.LastN(7)
    for u in users:
        udf = ml_ratings[ml_ratings.user == u]
        tst = samp(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) == 7
        assert len(tst) + len(trn) == len(udf)
        assert tst.timestamp.min() >= trn.timestamp.max()


def test_last_frac(ml_ratings: pd.DataFrame):
    users = np.random.choice(ml_ratings.user.unique(), 5, replace=False)

    samp = xf.LastFrac(0.2, "timestamp")
    for u in users:
        udf = ml_ratings[ml_ratings.user == u]
        tst = samp(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) + len(trn) == len(udf)
        assert len(tst) >= math.floor(len(udf) * 0.2)
        assert len(tst) <= math.ceil(len(udf) * 0.2)
        assert tst.timestamp.min() >= trn.timestamp.max()

    samp = xf.LastFrac(0.5, "timestamp")
    for u in users:
        udf = ml_ratings[ml_ratings.user == u]
        tst = samp(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) + len(trn) == len(udf)
        assert len(tst) >= math.floor(len(udf) * 0.5)
        assert len(tst) <= math.ceil(len(udf) * 0.5)
        assert tst.timestamp.min() >= trn.timestamp.max()


def test_partition_users(ml_ratings: pd.DataFrame):
    splits = xf.partition_users(ml_ratings, 5, xf.SampleN(5))
    splits = list(splits)
    assert len(splits) == 5

    for s in splits:
        ucounts = s.test.groupby("user").agg("count")
        assert all(ucounts == 5)
        assert all(s.test.index.union(s.train.index) == ml_ratings.index)
        assert all(s.train["user"].isin(s.train["user"].unique()))
        assert len(s.test) + len(s.train) == len(ml_ratings)

    users = ft.reduce(lambda us1, us2: us1 | us2, (set(s.test.user) for s in splits))
    assert len(users) == ml_ratings.user.nunique()
    assert users == set(ml_ratings.user)


def test_partition_may_skip_train(ml_ratings: pd.DataFrame):
    "Partitioning when users may not have enough ratings to be in the train and test sets."
    # make a data set where some users only have 1 rating
    ml_ratings = ml_ratings.sample(frac=0.1)
    users = ml_ratings.groupby("user")["rating"].count()
    assert users.min() == 1.0  # we should have some small users!
    users.name = "ur_count"

    splits = xf.partition_users(ml_ratings, 5, xf.SampleN(1))
    splits = list(splits)
    assert len(splits) == 5

    # now we go make sure we're missing some users! And don't have any NaN ml_ratings
    for train, test in splits:
        # no null ml_ratings
        assert all(train["rating"].notna())
        # see if test users with 1 rating are missing from train
        test = test.join(users, on="user")
        assert all(~(test.loc[test["ur_count"] == 1, "user"].isin(train["user"].unique())))
        # and users with more than one rating are in train
        assert all(test.loc[test["ur_count"] > 1, "user"].isin(train["user"].unique()))


def test_partition_users_frac(ml_ratings: pd.DataFrame):
    splits = xf.partition_users(ml_ratings, 5, xf.SampleFrac(0.2))
    splits = list(splits)
    assert len(splits) == 5
    ucounts = ml_ratings.groupby("user").item.count()
    uss = ucounts * 0.2

    for s in splits:
        tucs = s.test.groupby("user").item.count()
        assert all(tucs >= uss.loc[tucs.index] - 1)
        assert all(tucs <= uss.loc[tucs.index] + 1)
        assert all(s.test.index.union(s.train.index) == ml_ratings.index)
        assert len(s.test) + len(s.train) == len(ml_ratings)

    # we have all users
    users = ft.reduce(lambda us1, us2: us1 | us2, (set(s.test.user) for s in splits))
    assert len(users) == ml_ratings.user.nunique()
    assert users == set(ml_ratings.user)


def test_sample_users(ml_ratings: pd.DataFrame):
    splits = xf.sample_users(ml_ratings, 5, 100, xf.SampleN(5))
    splits = list(splits)
    assert len(splits) == 5

    for s in splits:
        ucounts = s.test.groupby("user").agg("count")
        assert len(s.test) == 5 * 100
        assert len(ucounts) == 100
        assert all(ucounts == 5)
        assert all(s.test.index.union(s.train.index) == ml_ratings.index)
        assert len(s.test) + len(s.train) == len(ml_ratings)

    # no overlapping users
    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue
        us1 = s1.test.user.unique()
        us2 = s2.test.user.unique()
        assert len(np.intersect1d(us1, us2)) == 0


def test_sample_users_frac(ml_ratings: pd.DataFrame):
    splits = xf.sample_users(ml_ratings, 5, 100, xf.SampleFrac(0.2))
    splits = list(splits)
    assert len(splits) == 5
    ucounts = ml_ratings.groupby("user").item.count()
    uss = ucounts * 0.2

    for s in splits:
        tucs = s.test.groupby("user").item.count()
        assert len(tucs) == 100
        assert all(tucs >= uss.loc[tucs.index] - 1)
        assert all(tucs <= uss.loc[tucs.index] + 1)
        assert all(s.test.index.union(s.train.index) == ml_ratings.index)
        assert len(s.test) + len(s.train) == len(ml_ratings)

    # no overlapping users
    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue
        us1 = s1.test.user.unique()
        us2 = s2.test.user.unique()
        assert len(np.intersect1d(us1, us2)) == 0


@pytest.mark.slow
def test_sample_users_frac_oversize(ml_ratings: pd.DataFrame):
    splits = xf.sample_users(ml_ratings, 20, 100, xf.SampleN(5))
    splits = list(splits)
    assert len(splits) == 20

    for s in splits:
        ucounts = s.test.groupby("user").agg("count")
        assert len(ucounts) < 100
        assert all(ucounts == 5)
        assert all(s.test.index.union(s.train.index) == ml_ratings.index)
        assert len(s.test) + len(s.train) == len(ml_ratings)

    users = ft.reduce(lambda us1, us2: us1 | us2, (set(s.test.user) for s in splits))
    assert len(users) == ml_ratings.user.nunique()
    assert users == set(ml_ratings.user)
    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue

        us1 = s1.test.user.unique()
        us2 = s2.test.user.unique()
        assert len(np.intersect1d(us1, us2)) == 0


def test_sample_users_frac_oversize_ndj(ml_ratings: pd.DataFrame):
    splits = xf.sample_users(ml_ratings, 20, 100, xf.SampleN(5), disjoint=False)
    splits = list(splits)
    assert len(splits) == 20

    for s in splits:
        ucounts = s.test.groupby("user").agg("count")
        assert len(ucounts) == 100
        assert len(s.test) == 5 * 100
        assert all(ucounts == 5)
        assert all(s.test.index.union(s.train.index) == ml_ratings.index)
        assert len(s.test) + len(s.train) == len(ml_ratings)


def test_non_unique_index_partition_users(ml_ratings: pd.DataFrame):
    """Partitioning users when dataframe has non-unique indices"""
    ml_ratings = ml_ratings.set_index("user")  ##forces non-unique index
    with pytest.raises(ValueError):
        for split in xf.partition_users(ml_ratings, 5, xf.SampleN(5)):
            pass


def test_sample_users_dup_index(ml_ratings: pd.DataFrame):
    """Sampling users when dataframe has non-unique indices"""
    ml_ratings = ml_ratings.set_index("user")  ##forces non-unique index
    with pytest.raises(ValueError):
        for split in xf.sample_users(ml_ratings, 5, 100, xf.SampleN(5)):
            pass


def test_sample_rows_dup_index(ml_ratings: pd.DataFrame):
    """Sampling ml_ratings when dataframe has non-unique indices"""
    ml_ratings = ml_ratings.set_index("user")  ##forces non-unique index
    with pytest.raises(ValueError):
        for split in xf.sample_rows(ml_ratings, partitions=5, size=1000):
            pass


def test_partition_users_dup_index(ml_ratings: pd.DataFrame):
    """Partitioning ml_ratings when dataframe has non-unique indices"""
    ml_ratings = ml_ratings.set_index("user")  ##forces non-unique index
    with pytest.raises(ValueError):
        for split in xf.partition_users(ml_ratings, 5, xf.SampleN(5)):
            pass
