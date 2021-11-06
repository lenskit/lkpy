import itertools as it
import functools as ft
import pytest
import math

import numpy as np

import lenskit.util.test as lktu

import lenskit.crossfold as xf


def test_partition_rows():
    ratings = lktu.ml_test.ratings
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
    ratings = lktu.ml_test.ratings
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
    ratings = lktu.ml_test.ratings
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
    ratings = lktu.ml_test.ratings
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
    ratings = lktu.ml_test.ratings
    splits = xf.sample_rows(ratings, 150, 1000)
    splits = list(splits)
    assert len(splits) == 150

    for s in splits:
        assert len(s.test) + len(s.train) == len(ratings)
        assert all(s.test.index.union(s.train.index) == ratings.index)
        test_idx = s.test.set_index(['user', 'item']).index
        train_idx = s.train.set_index(['user', 'item']).index
        assert len(test_idx.intersection(train_idx)) == 0


def test_sample_n():
    ratings = lktu.ml_test.ratings

    users = np.random.choice(ratings.user.unique(), 5, replace=False)

    s5 = xf.SampleN(5)
    for u in users:
        udf = ratings[ratings.user == u]
        tst = s5(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) == 5
        assert len(tst) + len(trn) == len(udf)

    s10 = xf.SampleN(10)
    for u in users:
        udf = ratings[ratings.user == u]
        tst = s10(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) == 10
        assert len(tst) + len(trn) == len(udf)


def test_sample_frac():
    ratings = lktu.ml_test.ratings
    users = np.random.choice(ratings.user.unique(), 5, replace=False)

    samp = xf.SampleFrac(0.2)
    for u in users:
        udf = ratings[ratings.user == u]
        tst = samp(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) + len(trn) == len(udf)
        assert len(tst) >= math.floor(len(udf) * 0.2)
        assert len(tst) <= math.ceil(len(udf) * 0.2)

    samp = xf.SampleFrac(0.5)
    for u in users:
        udf = ratings[ratings.user == u]
        tst = samp(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) + len(trn) == len(udf)
        assert len(tst) >= math.floor(len(udf) * 0.5)
        assert len(tst) <= math.ceil(len(udf) * 0.5)


def test_last_n():
    ratings = lktu.ml_test.ratings
    users = np.random.choice(ratings.user.unique(), 5, replace=False)

    samp = xf.LastN(5)
    for u in users:
        udf = ratings[ratings.user == u]
        tst = samp(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) == 5
        assert len(tst) + len(trn) == len(udf)
        assert tst.timestamp.min() >= trn.timestamp.max()

    samp = xf.LastN(7)
    for u in users:
        udf = ratings[ratings.user == u]
        tst = samp(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) == 7
        assert len(tst) + len(trn) == len(udf)
        assert tst.timestamp.min() >= trn.timestamp.max()


def test_last_frac():
    ratings = lktu.ml_test.ratings
    users = np.random.choice(ratings.user.unique(), 5, replace=False)

    samp = xf.LastFrac(0.2, 'timestamp')
    for u in users:
        udf = ratings[ratings.user == u]
        tst = samp(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) + len(trn) == len(udf)
        assert len(tst) >= math.floor(len(udf) * 0.2)
        assert len(tst) <= math.ceil(len(udf) * 0.2)
        assert tst.timestamp.min() >= trn.timestamp.max()

    samp = xf.LastFrac(0.5, 'timestamp')
    for u in users:
        udf = ratings[ratings.user == u]
        tst = samp(udf)
        trn = udf.loc[udf.index.difference(tst.index), :]
        assert len(tst) + len(trn) == len(udf)
        assert len(tst) >= math.floor(len(udf) * 0.5)
        assert len(tst) <= math.ceil(len(udf) * 0.5)
        assert tst.timestamp.min() >= trn.timestamp.max()


def test_partition_users():
    ratings = lktu.ml_test.ratings
    splits = xf.partition_users(ratings, 5, xf.SampleN(5))
    splits = list(splits)
    assert len(splits) == 5

    for s in splits:
        ucounts = s.test.groupby('user').agg('count')
        assert all(ucounts == 5)
        assert all(s.test.index.union(s.train.index) == ratings.index)
        assert all(s.train['user'].isin(s.train['user'].unique()))
        assert len(s.test) + len(s.train) == len(ratings)

    users = ft.reduce(lambda us1, us2: us1 | us2,
                      (set(s.test.user) for s in splits))
    assert len(users) == ratings.user.nunique()
    assert users == set(ratings.user)


def test_partition_may_skip_train():
    "Partitioning when users may not have enough ratings to be in the train set and test set."
    ratings = lktu.ml_test.ratings
    # make a data set where some users only have 1 rating
    ratings = ratings.sample(frac=0.1)
    users = ratings.groupby('user')['rating'].count()
    assert users.min() == 1.0  # we should have some small users!
    users.name = 'ur_count'

    splits = xf.partition_users(ratings, 5, xf.SampleN(1))
    splits = list(splits)
    assert len(splits) == 5

    # now we go make sure we're missing some users! And don't have any NaN ratings
    for train, test in splits:
        # no null ratings
        assert all(train['rating'].notna())
        # see if test users with 1 rating are missing from train
        test = test.join(users, on='user')
        assert all(~(test.loc[test['ur_count'] == 1, 'user'].isin(train['user'].unique())))
        # and users with more than one rating are in train
        assert all(test.loc[test['ur_count'] > 1, 'user'].isin(train['user'].unique()))


def test_partition_users_frac():
    ratings = lktu.ml_test.ratings
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
    ratings = lktu.ml_test.ratings
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
    ratings = lktu.ml_test.ratings
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


@pytest.mark.slow
def test_sample_users_frac_oversize():
    ratings = lktu.ml_test.ratings
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
    ratings = lktu.ml_test.ratings
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

def test_non_unique_index_partition_users():
    ratings = lktu.ml_test.ratings
    ratings = ratings.set_index('user')  ##forces non-unique index
    with pytest.raises(ValueError):
        splits = xf.partition_users(ratings, 5, xf.SampleN(5))
        splits = list(splits)

