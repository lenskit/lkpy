import itertools as it
import functools as ft
import pytest

import numpy as np

import lk_test_utils as lktu

import lenskit.crossfold as xf


def test_partition_rows():
    ratings = lktu.ml_pandas.renamed.ratings
    splits = xf.partition_rows(ratings, 5)
    splits = list(splits)
    assert len(splits) == 5

    for s in splits:
        assert len(s.test) + len(s.train) == len(ratings)
        assert all(s.test.index.union(s.train.index) == ratings.index)
        test_idx = s.test.set_index(['user', 'item']).index
        train_idx = s.train.set_index(['user', 'item']).index
        assert len(test_idx.intersection(train_idx)) == 0

    # we should partition!
    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue

        i1 = s1.test.set_index(['user', 'item']).index
        i2 = s2.test.set_index(['user', 'item']).index
        inter = i1.intersection(i2)
        assert len(inter) == 0

    union = ft.reduce(lambda i1, i2: i1.union(i2), (s.test.index for s in splits))
    assert len(union.unique()) == len(ratings)


def test_sample_rows():
    ratings = lktu.ml_pandas.renamed.ratings
    splits = xf.sample_rows(ratings, partitions=5, size=1000)
    splits = list(splits)
    assert len(splits) == 5

    for s in splits:
        assert len(s.test) == 1000
        assert len(s.test) + len(s.train) == len(ratings)
        test_idx = s.test.set_index(['user', 'item']).index
        train_idx = s.train.set_index(['user', 'item']).index
        assert len(test_idx.intersection(train_idx)) == 0

    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue

        i1 = s1.test.set_index(['user', 'item']).index
        i2 = s2.test.set_index(['user', 'item']).index
        inter = i1.intersection(i2)
        assert len(inter) == 0


def test_sample_rows_more_smaller_parts():
    ratings = lktu.ml_pandas.renamed.ratings
    splits = xf.sample_rows(ratings, partitions=10, size=500)
    splits = list(splits)
    assert len(splits) == 10

    for s in splits:
        assert len(s.test) == 500
        assert len(s.test) + len(s.train) == len(ratings)
        test_idx = s.test.set_index(['user', 'item']).index
        train_idx = s.train.set_index(['user', 'item']).index
        assert len(test_idx.intersection(train_idx)) == 0

    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue

        i1 = s1.test.set_index(['user', 'item']).index
        i2 = s2.test.set_index(['user', 'item']).index
        inter = i1.intersection(i2)
        assert len(inter) == 0


def test_sample_non_disjoint():
    ratings = lktu.ml_pandas.renamed.ratings
    splits = xf.sample_rows(ratings, partitions=10, size=1000, disjoint=False)
    splits = list(splits)
    assert len(splits) == 10

    for s in splits:
        assert len(s.test) == 1000
        assert len(s.test) + len(s.train) == len(ratings)
        test_idx = s.test.set_index(['user', 'item']).index
        train_idx = s.train.set_index(['user', 'item']).index
        assert len(test_idx.intersection(train_idx)) == 0

    # There are enough splits & items we should pick at least one duplicate
    ipairs = ((s1.test.set_index('user', 'item').index, s2.test.set_index('user', 'item').index)
              for (s1, s2) in it.product(splits, splits))
    isizes = [len(i1.intersection(i2)) for (i1, i2) in ipairs]
    assert any(n > 0 for n in isizes)


@pytest.mark.slow
def test_sample_oversize():
    ratings = lktu.ml_pandas.renamed.ratings
    splits = xf.sample_rows(ratings, 150, 1000)
    splits = list(splits)
    assert len(splits) == 150

    for s in splits:
        assert len(s.test) + len(s.train) == len(ratings)
        assert all(s.test.index.union(s.train.index) == ratings.index)
        test_idx = s.test.set_index(['user', 'item']).index
        train_idx = s.train.set_index(['user', 'item']).index
        assert len(test_idx.intersection(train_idx)) == 0


@pytest.mark.skip(reason='Not yet working')
def test_sample_dask():
    ratings = lktu.ml_dask.renamed.ratings
    splits = xf.sample_rows(ratings, partitions=5, size=1000)
    splits = list(splits)
    assert len(splits) == 5

    for s in splits:
        assert len(s.test) == 1000
        assert len(s.test) + len(s.train) == len(ratings)
        test_idx = s.test.set_index(['user', 'item']).index
        train_idx = s.train.set_index(['user', 'item']).index
        assert len(test_idx.intersection(train_idx)) == 0

    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue

        i1 = s1.test.set_index(['user', 'item']).index
        i2 = s2.test.set_index(['user', 'item']).index
        inter = i1.intersection(i2)
        assert len(inter) == 0


@pytest.mark.skip(reason='Not yet working')
def test_partition_dask():
    ratings = lktu.ml_dask.renamed.ratings
    splits = xf.partition_rows(ratings, 5)
    splits = list(splits)
    assert len(splits) == 5

    for s in splits:
        assert len(s.test) + len(s.train) == len(ratings)
        assert all(s.test.index.union(s.train.index) == ratings.index)
        test_idx = s.test.set_index(['user', 'item']).index
        train_idx = s.train.set_index(['user', 'item']).index
        assert len(test_idx.intersection(train_idx)) == 0


def test_partition_users():
    ratings = lktu.ml_pandas.renamed.ratings
    splits = xf.partition_users(ratings, 5, xf.SampleN(5))
    splits = list(splits)
    assert len(splits) == 5

    for s in splits:
        ucounts = s.test.groupby('user').agg('count')
        assert all(ucounts == 5)
        assert all(s.test.index.union(s.train.index) == ratings.index)
        assert len(s.test) + len(s.train) == len(ratings)

    users = ft.reduce(lambda us1, us2: us1 | us2,
                      (set(s.test.user) for s in splits))
    assert len(users) == ratings.user.nunique()
    assert users == set(ratings.user)


def test_partition_users_frac():
    ratings = lktu.ml_pandas.renamed.ratings
    splits = xf.partition_users(ratings, 5, xf.SampleFrac(0.2))
    splits = list(splits)
    assert len(splits) == 5
    ucounts = ratings.groupby('user').item.count()
    uss = ucounts * 0.2

    for s in splits:
        tucs = s.test.groupby('user').item.count()
        assert all(tucs >= uss.loc[tucs.index] - 1)
        assert all(tucs <= uss.loc[tucs.index] + 1)
        assert all(s.test.index.union(s.train.index) == ratings.index)
        assert len(s.test) + len(s.train) == len(ratings)

    # we have all users
    users = ft.reduce(lambda us1, us2: us1 | us2,
                      (set(s.test.user) for s in splits))
    assert len(users) == ratings.user.nunique()
    assert users == set(ratings.user)


def test_sample_users():
    ratings = lktu.ml_pandas.renamed.ratings
    splits = xf.sample_users(ratings, 5, 100, xf.SampleN(5))
    splits = list(splits)
    assert len(splits) == 5

    for s in splits:
        ucounts = s.test.groupby('user').agg('count')
        assert len(s.test) == 5 * 100
        assert len(ucounts) == 100
        assert all(ucounts == 5)
        assert all(s.test.index.union(s.train.index) == ratings.index)
        assert len(s.test) + len(s.train) == len(ratings)

    # no overlapping users
    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue
        us1 = s1.test.user.unique()
        us2 = s2.test.user.unique()
        assert len(np.intersect1d(us1, us2)) == 0


def test_sample_users_frac():
    ratings = lktu.ml_pandas.renamed.ratings
    splits = xf.sample_users(ratings, 5, 100, xf.SampleFrac(0.2))
    splits = list(splits)
    assert len(splits) == 5
    ucounts = ratings.groupby('user').item.count()
    uss = ucounts * 0.2

    for s in splits:
        tucs = s.test.groupby('user').item.count()
        assert len(tucs) == 100
        assert all(tucs >= uss.loc[tucs.index] - 1)
        assert all(tucs <= uss.loc[tucs.index] + 1)
        assert all(s.test.index.union(s.train.index) == ratings.index)
        assert len(s.test) + len(s.train) == len(ratings)

    # no overlapping users
    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue
        us1 = s1.test.user.unique()
        us2 = s2.test.user.unique()
        assert len(np.intersect1d(us1, us2)) == 0


def test_sample_users_frac_oversize():
    ratings = lktu.ml_pandas.renamed.ratings
    splits = xf.sample_users(ratings, 20, 100, xf.SampleN(5))
    splits = list(splits)
    assert len(splits) == 20

    for s in splits:
        ucounts = s.test.groupby('user').agg('count')
        assert len(ucounts) < 100
        assert all(ucounts == 5)
        assert all(s.test.index.union(s.train.index) == ratings.index)
        assert len(s.test) + len(s.train) == len(ratings)

    users = ft.reduce(lambda us1, us2: us1 | us2,
                      (set(s.test.user) for s in splits))
    assert len(users) == ratings.user.nunique()
    assert users == set(ratings.user)
    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue

        us1 = s1.test.user.unique()
        us2 = s2.test.user.unique()
        assert len(np.intersect1d(us1, us2)) == 0


def test_sample_users_frac_oversize_ndj():
    ratings = lktu.ml_pandas.renamed.ratings
    splits = xf.sample_users(ratings, 20, 100, xf.SampleN(5), disjoint=False)
    splits = list(splits)
    assert len(splits) == 20

    for s in splits:
        ucounts = s.test.groupby('user').agg('count')
        assert len(ucounts) == 100
        assert len(s.test) == 5 * 100
        assert all(ucounts == 5)
        assert all(s.test.index.union(s.train.index) == ratings.index)
        assert len(s.test) + len(s.train) == len(ratings)
