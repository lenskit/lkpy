# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import functools as ft
import itertools as it
import math

import numpy as np
import pandas as pd

import pytest

from lenskit.data import Dataset, from_interactions_df
from lenskit.splitting.holdout import SampleFrac, SampleN
from lenskit.splitting.users import crossfold_users, sample_users


def test_crossfold_users(ml_ds: Dataset):
    splits = crossfold_users(ml_ds, 5, SampleN(5))
    splits = list(splits)
    assert len(splits) == 5

    users = set()
    for s in splits:
        assert all(len(il) for il in s.test.lists())
        assert not any(u in users for u in s.test.keys())
        users |= set(k.user_id for k in s.test.keys())

        test_pairs = set((u, i) for (u, il) in s.test for i in il.ids())
        tdf = s.train.interaction_matrix(format="pandas", field="rating", original_ids=True)
        train_pairs = set(zip(tdf["user_id"], tdf["item_id"]))
        assert not test_pairs & train_pairs
        assert s.test_size + s.train.interactions().count() == ml_ds.interactions().count()

    assert users == set(ml_ds.users)


def test_crossfold_may_skip_train(ml_ratings: pd.DataFrame):
    "Partitioning when users may not have enough ratings to be in the train and test sets."
    # make a data set where some users only have 1 rating
    ml_ratings = ml_ratings.sample(frac=0.1)
    ucounts = ml_ratings.groupby("user_id")["rating"].count()
    assert ucounts.min() == 1  # we should have some small users!
    ucounts.name = "ur_count"
    ml_ds = from_interactions_df(ml_ratings)

    splits = crossfold_users(ml_ds, 5, SampleN(1))
    splits = list(splits)
    assert len(splits) == 5

    # now we go make sure we're missing some users! And don't have any NaN ml_ratings
    for split in splits:
        for u in ucounts[ucounts == 1].index:
            if u in split.test:
                row = split.train.user_row(u)
                assert row is not None
                assert len(row) == 0


def test_crossfold_users_frac(ml_ds: Dataset):
    splits = crossfold_users(ml_ds, 5, SampleFrac(0.2))
    splits = list(splits)
    assert len(splits) == 5
    ustats = ml_ds.user_stats()
    uss = ustats["count"] * 0.2

    for s in splits:
        assert all(len(il) >= uss.loc[u.user_id] - 1 for (u, il) in s.test)
        assert all(len(il) <= uss.loc[u.user_id] + 1 for (u, il) in s.test)
        assert s.test_size + s.train.interactions().count() == ml_ds.interactions().count()


def test_sample_users_single(ml_ds: Dataset):
    split = sample_users(ml_ds, 100, SampleN(5))

    assert len(split.test) == 100
    assert split.test_size == 500

    test_pairs = set((u, i) for (u, il) in split.test for i in il.ids())
    assert len(test_pairs) == split.test_size
    tdf = split.train.interaction_matrix(format="pandas", field="rating", original_ids=True)
    train_pairs = set(zip(tdf["user_id"], tdf["item_id"]))
    assert len(train_pairs) == split.train.interactions().count()
    assert len(test_pairs & train_pairs) == 0
    assert split.test_size + split.train.interactions().count() == ml_ds.interactions().count()


def test_sample_users(ml_ds: Dataset):
    splits = sample_users(ml_ds, 100, SampleN(5), repeats=5)
    splits = list(splits)
    assert len(splits) == 5

    aus = set()
    for s in splits:
        assert len(s.test) == 100
        assert s.test_size == 500
        # users are disjoint
        assert not any(u in aus for u in s.test.keys())
        aus |= set(k.user_id for k in s.test.keys())

        test_pairs = set((u, i) for (u, il) in s.test for i in il.ids())
        assert len(test_pairs) == s.test_size
        tdf = s.train.interaction_matrix(format="pandas", field="rating", original_ids=True)
        train_pairs = set(zip(tdf["user_id"], tdf["item_id"]))
        assert len(train_pairs) == s.train.interactions().count()
        assert len(test_pairs & train_pairs) == 0
        assert s.test_size + s.train.interactions().count() == ml_ds.interactions().count()


def test_sample_test_only(ml_ds: Dataset):
    splits = sample_users(ml_ds, 100, SampleN(5), repeats=1, test_only=True)
    splits = list(splits)
    assert len(splits) == 1

    for s in splits:
        assert len(s.test) == 100
        assert s.test_size == 500
        assert s.train.interaction_count == 0


def test_sample_users_non_disjoint(ml_ds: Dataset):
    splits = sample_users(ml_ds, 100, SampleN(5), repeats=5, disjoint=False)
    splits = list(splits)
    assert len(splits) == 5

    aus = set()

    for s in splits:
        assert len(s.test) == 100
        assert s.test_size == 500
        aus |= set(k.user_id for k in s.test.keys())

        test_pairs = set((u, i) for (u, il) in s.test for i in il.ids())
        assert len(test_pairs) == s.test_size
        tdf = s.train.interaction_matrix(format="pandas", field="rating", original_ids=True)
        train_pairs = set(zip(tdf["user_id"], tdf["item_id"]))
        assert len(train_pairs) == s.train.interactions().count()
        assert len(test_pairs & train_pairs) == 0
        assert s.test_size + s.train.interactions().count() == ml_ds.interactions().count()

    # some user appears at least once
    assert len(aus) < 500


@pytest.mark.slow
def test_sample_users_frac_oversize(ml_ds: Dataset):
    splits = sample_users(ml_ds, 100, SampleN(5), repeats=20)
    splits = list(splits)
    assert len(splits) == 20

    for s in splits:
        assert len(s.test) < 100
        assert all(len(il) == 5 for il in s.test.lists())


def test_sample_users_frac_oversize_ndj(ml_ds: Dataset):
    splits = sample_users(ml_ds, 100, SampleN(5), repeats=20, disjoint=False)
    splits = list(splits)
    assert len(splits) == 20

    for s in splits:
        assert len(s.test) == 100
        assert s.test_size == 5 * 100
        assert all([len(il) for il in s.test.lists()])
