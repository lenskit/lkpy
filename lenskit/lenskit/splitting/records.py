# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from typing import Iterator, overload

import numpy as np
import pandas as pd

from lenskit.data import Dataset, ItemListCollection, UserIDKey
from lenskit.data.builder import DatasetBuilder
from lenskit.random import RNGInput, random_generator

from .split import TTSplit

_log = logging.getLogger(__name__)


def crossfold_records(
    data: Dataset, partitions: int, *, test_only: bool = False, rng: RNGInput = None
) -> Iterator[TTSplit]:
    """
    Partition a dataset by **records** into cross-fold partitions.  This
    partitions the records (ratings, play counts, clicks, etc.) into *k*
    partitions without regard to users or items.

    Since record-based random cross-validation doesn't make much sense with
    repeated interactions, this splitter only supports operating on the
    dataset's interaction matrix.

    Stability:
        Caller

    Args:
        data:
            Ratings or other data you wish to partition.
        partitions:
            The number of partitions to produce.
        test_only:
            If ``True``, returns splits with empty training sets (useful when
            you just want to save the test data).
        rng:
            The random number generator or seed (see :ref:`rng`).

    Returns:
        iterator: an iterator of train-test pairs
    """

    _log.info("partitioning %d ratings into %d partitions", data.interactions().count(), partitions)
    rng = random_generator(rng)

    # get the full data list to split
    df = data.interactions().pandas(ids=True)
    n = len(df)
    rows = np.arange(n)

    # shuffle the indices & split into partitions
    rng.shuffle(rows)
    test_sets = np.array_split(rows, partitions)

    # convert each partition into a split
    for ts in test_sets:
        yield _make_pair(data, df, ts, test_only=test_only)


@overload
def sample_records(
    data: Dataset,
    size: int,
    *,
    disjoint: bool = True,
    test_only: bool = False,
    rng: RNGInput = None,
    repeats: None = None,
) -> TTSplit: ...
@overload
def sample_records(
    data: Dataset,
    size: int,
    *,
    repeats: int,
    disjoint: bool = True,
    test_only: bool = False,
    rng: RNGInput = None,
) -> Iterator[TTSplit]: ...
def sample_records(
    data: Dataset,
    size: int,
    *,
    repeats: int | None = None,
    disjoint: bool = True,
    test_only: bool = False,
    rng: RNGInput = None,
) -> TTSplit | Iterator[TTSplit]:
    """
    Sample train-test a frame of ratings into train-test partitions.  This
    function does not care what kind of data is in `data`, so long as it is a
    Pandas DataFrame (or equivalent).

    We can loop over a sequence of train-test pairs::

        >>> from lenskit.data import load_movielens
        >>> movielens = load_movielens('data/ml-latest-small')
        >>> for split in sample_records(movielens, 1000, repeats=5):
        ...     print(sum(len(il) for il in split.test.lists()))
        1000
        1000
        1000
        1000
        1000

    Sometimes for testing, it is useful to just get a single pair::

        >>> split = sample_records(movielens, 1000)
        >>> sum(len(il) for il in split.test.lists())
        1000

    Stability:
        Caller

    Args:
        data:
            The data set to split.
        size:
            The size of each test sample.
        repeats:
            The number of data splits to produce.  If ``None``, produce a
            _single_ train-test pair instead of an iterator or list.
        disjoint:
            If ``True``, force test samples to be disjoint.
        test_only:
            If ``True``, returns splits with empty training sets (useful when
            you just want to save the test data).
        rng:
            The random number generator or seed (see :ref:`rng`).

    Returns:
        A train-test pair or iterator of such pairs (depending on ``repeats``).
    """

    rng = random_generator(rng)

    # get the full data list to split
    df = data.interactions().pandas(ids=True)
    n = len(df)

    if repeats is None:
        test_pos = rng.choice(np.int32(n), size, replace=False)
        return _make_pair(data, df, test_pos)

    if disjoint and repeats * size >= n:
        _log.warning(
            "wanted %d disjoint splits of %d each, but only have %d rows; cross-folding",
            repeats,
            size,
            n,
        )
        return crossfold_records(data, repeats, rng=rng)

    # get iterators over index arrays for producing the data
    if disjoint:
        _log.info("creating %d disjoint samples of size %d", repeats, size)
        ips = _disjoint_samples(n, size, repeats, rng)

    else:
        _log.info("taking %d samples of size %d", repeats, size)
        ips = _n_samples(n, size, repeats, rng)

    # since this func is both generator and return depending on args,
    # we can't use yield — need to return a generator expression
    return (_make_pair(data, df, test_is, test_only=test_only) for test_is in ips)


def _make_pair(
    data: Dataset,
    df: pd.DataFrame,
    test_is: np.ndarray[int, np.dtype[np.int32]],
    *,
    test_only: bool = False,
) -> TTSplit:
    mask = np.zeros(len(df), np.bool_)
    mask[test_is] = True

    test = ItemListCollection.from_df(df[mask], UserIDKey)
    train_build = DatasetBuilder(data)
    iname = data.default_interaction_class()
    train_build.clear_relationships(iname)
    if not test_only:
        train_build.add_interactions(iname, df[~mask])
    train = train_build.build()

    return TTSplit(train, test)


def _disjoint_samples(
    n: int, size: int, reps: int, rng: np.random.Generator
) -> Iterator[np.ndarray[int, np.dtype[np.int32]]]:
    # shuffle the indices & split into partitions
    xs = np.arange(n, dtype=np.int32)
    rng.shuffle(xs)

    # convert each partition into a split
    for i in range(reps):
        start = i * size
        end = start + size
        yield xs[start:end]


def _n_samples(
    n: int, size: int, reps: int, rng: np.random.Generator
) -> Iterator[np.ndarray[int, np.dtype[np.int32]]]:
    for i in range(reps):
        yield rng.choice(np.int32(n), size, replace=False)
