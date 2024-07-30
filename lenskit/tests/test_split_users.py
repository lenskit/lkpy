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
