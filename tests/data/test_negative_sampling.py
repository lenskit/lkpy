# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np

from pytest import mark

from lenskit.data import Dataset
from lenskit.logging import get_logger

_log = get_logger(__name__)


@mark.parametrize("weighting", ["uniform", "popularity"])
def test_negative(rng: np.random.Generator, ml_ds: Dataset, weighting):
    log = _log.bind()
    matrix = ml_ds.interactions().matrix()

    users = rng.choice(ml_ds.user_count, 100, replace=True)
    users = np.require(users, "i4")

    negatives = matrix.sample_negatives(users, rng=rng, weighting=weighting)
    assert negatives.shape == (100,)

    log.info("checking basic item results")
    assert np.all(negatives >= 0)
    assert np.all(negatives < ml_ds.item_count)
    log.info("checking negative items")
    for u, i in zip(users, negatives):
        ulog = log.bind(
            user_num=u.item(),
            user_id=int(ml_ds.users.id(u)),  # type: ignore
            item_num=i.item(),
            item_id=int(ml_ds.items.id(i)),  # type: ignore
        )
        row = ml_ds.user_row(user_num=u)
        ulog = ulog.bind(u_nitems=len(row))
        ulog.debug("checking if item is negative")
        assert (u, i) not in matrix.rc_index
        print(ml_ds.users.id(u), row.ids())
        assert i not in row.numbers()


@mark.parametrize("weighting", ["uniform", "popularity"])
def test_negative_multiple(rng: np.random.Generator, ml_ds: Dataset, weighting):
    log = _log.bind()
    matrix = ml_ds.interactions().matrix()

    users = rng.choice(ml_ds.user_count, 100, replace=True)
    users = np.require(users, "i4")

    negatives = matrix.sample_negatives(users, rng=rng, weighting=weighting, n=2)
    assert negatives.shape == (100, 2)

    log.info("checking basic item results")
    assert np.all(negatives >= 0)
    assert np.all(negatives < ml_ds.item_count)
    log.info("checking negative items")
    for c in range(2):
        for u, i in zip(users, negatives[:, c]):
            ulog = log.bind(
                user_num=u.item(),
                user_id=int(ml_ds.users.id(u)),  # type: ignore
                item_num=i.item(),
                item_id=int(ml_ds.items.id(i)),  # type: ignore
            )
            row = ml_ds.user_row(user_num=u)
            ulog = ulog.bind(u_nitems=len(row))
            ulog.debug("checking if item is negative")
            assert (u, i) not in matrix.rc_index
            print(ml_ds.users.id(u), row.ids())
            assert i not in row.numbers()


@mark.parametrize("weighting", ["uniform", "popularity"])
def test_negative_unverified(rng: np.random.Generator, ml_ds: Dataset, weighting):
    matrix = ml_ds.interactions().matrix()

    users = rng.choice(ml_ds.user_count, 500, replace=True)
    users = np.require(users, "i4")

    negatives = matrix.sample_negatives(users, verify=False, rng=rng, weighting=weighting)

    assert np.all(negatives >= 0)
    assert np.all(negatives < ml_ds.item_count)


@mark.benchmark()
def test_negative_unverified_bench(rng: np.random.Generator, ml_ds: Dataset, benchmark):
    matrix = ml_ds.interactions().matrix()

    users = rng.choice(ml_ds.user_count, 500, replace=True)
    users = np.require(users, "i4")

    def sample():
        _items = matrix.sample_negatives(users, verify=False, rng=rng)

    benchmark(sample)


@mark.benchmark()
def test_negative_verified_bench(rng: np.random.Generator, ml_ds: Dataset, benchmark):
    matrix = ml_ds.interactions().matrix()

    users = rng.choice(ml_ds.user_count, 500, replace=True)
    users = np.require(users, "i4")

    def sample():
        _items = matrix.sample_negatives(users, rng=rng)

    benchmark(sample)


@mark.benchmark()
def test_negative_5000_bench(rng: np.random.Generator, ml_ds: Dataset, benchmark):
    matrix = ml_ds.interactions().matrix()

    users = rng.choice(ml_ds.user_count, 5000, replace=True)
    users = np.require(users, "i4")

    def sample():
        _items = matrix.sample_negatives(users, rng=rng)

    benchmark(sample)


@mark.benchmark()
def test_negative_multiple_bench(rng: np.random.Generator, ml_ds: Dataset, benchmark):
    matrix = ml_ds.interactions().matrix()

    users = rng.choice(ml_ds.user_count, 500, replace=True)
    users = np.require(users, "i4")

    def sample():
        _items = matrix.sample_negatives(users, n=10, rng=rng)

    benchmark(sample)
