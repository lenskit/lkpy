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
from numpy.typing import NDArray

from lenskit.data import NPID, Dataset, ItemListCollection, UserIDKey
from lenskit.data.matrix import MatrixDataset
from lenskit.logging import item_progress
from lenskit.util.random import RNGInput, random_generator

from .holdout import HoldoutMethod
from .split import TTSplit

_log = logging.getLogger(__name__)


def crossfold_users(
    data: Dataset,
    partitions: int,
    method: HoldoutMethod,
    *,
    test_only: bool = False,
    rng: RNGInput | None = None,
) -> Iterator[TTSplit]:
    """
    Partition a frame of ratings or other data into train-test partitions
    user-by-user. This function does not care what kind of data is in `data`, so
    long as it is a Pandas DataFrame (or equivalent) and has a `user` column.

    Stability:
        Caller

    Args:
        data:
            a data frame containing ratings or other data you wish to partition.
        partitions:
            the number of partitions to produce
        method:
            The method for selecting test rows for each user.
        test_only:
            If ``True``, returns splits with only testing data.
        rng:
            The random number generator or seed (see :ref:`rng`).

    Returns
        The train-test pairs.
    """
    rng = random_generator(rng)

    users = data.users.ids()
    _log.info(
        "partitioning %d rows for %d users into %d partitions",
        data.count("pairs"),
        len(users),
        partitions,
    )

    # create an array of indexes into user row
    rows = np.arange(len(users))
    # shuffle the indices & split into partitions
    rng.shuffle(rows)
    test_sets = np.array_split(rows, partitions)

    # get the whole test DF
    df = data.interaction_matrix("pandas", field="all", original_ids=True).set_index(
        ["user_id", "item_id"]
    )

    # convert each partition into a split
    for i, ts in enumerate(test_sets):
        # get our users!
        test_us = users[ts]
        _log.info("fold %d: selecting test ratings", i)

        yield _make_split(data, df, test_us, method, test_only=test_only)


@overload
def sample_users(
    data: Dataset,
    size: int,
    method: HoldoutMethod,
    *,
    repeats: int,
    disjoint: bool = True,
    test_only: bool = False,
    rng: RNGInput = None,
) -> Iterator[TTSplit]: ...
@overload
def sample_users(
    data: Dataset,
    size: int,
    method: HoldoutMethod,
    *,
    disjoint: bool = True,
    rng: RNGInput = None,
    test_only: bool = False,
    repeats: None = None,
) -> TTSplit: ...
def sample_users(
    data: Dataset,
    size: int,
    method: HoldoutMethod,
    *,
    repeats: int | None = None,
    disjoint: bool = True,
    test_only: bool = False,
    rng: RNGInput = None,
) -> Iterator[TTSplit] | TTSplit:
    """
    Create train-test splits by sampling users.  When ``repeats`` is None,
    returns a single train-test split; otherwise, it returns an iterator over
    multiple splits. If ``repeats=1``, this function returns an iterator that
    yields a single train-test pair.

    Stability:
        Caller

    Args:
        data:
            Data frame containing ratings or other data you wish to partition.
        size:
            The sample size.
        method:
            The method for obtaining user test ratings.
        repeats:
            The number of samples to produce.
        test_only:
            If ``True``, returns splits with empty training sets (useful when
            you just want to save the test data).
        rng:
            The random number generator or seed (see :ref:`rng`).

    Returns:
        The train-test pair(s).
    """

    rng = random_generator(rng)

    users = data.users.ids()
    unums = np.arange(len(users))
    if disjoint and repeats is not None and repeats * size >= len(users):
        _log.warning(
            "cannot take %d disjoint samples of size %d from %d users", repeats, size, len(users)
        )
        return crossfold_users(data, repeats, method)

    _log.info("sampling %d users (n=%d)", len(users), size)

    # get the whole test DF
    rate_df = data.interaction_matrix("pandas", field="all", original_ids=True).set_index(
        ["user_id", "item_id"]
    )

    if repeats is None:
        test_us = rng.choice(users, size, replace=False)
        return _make_split(data, rate_df, test_us, method)

    if disjoint:
        rng.shuffle(unums)
        test_usets = [unums[i * size : (i + 1) * size] for i in range(repeats)]
    else:
        test_usets = [rng.choice(len(users), size, replace=False) for _i in range(repeats)]

    return (_make_split(data, rate_df, users[us], method, test_only=test_only) for us in test_usets)


def _make_split(
    data: Dataset,
    df: pd.DataFrame,
    test_us: NDArray[NPID],
    method: HoldoutMethod,
    *,
    test_only: bool = False,
) -> TTSplit:
    # create the test sets for these users
    test = ItemListCollection(UserIDKey)

    with item_progress("Splitting users", len(test_us)) as pb:
        for u in test_us:
            row = data.user_row(u)
            assert row is not None
            u_test = method(row)
            test.add(u_test, u)
            pb.update()

    if test_only:
        return TTSplit(
            MatrixDataset(data.users, data.items, pd.DataFrame({"user_id": [], "item_id": []})),
            test,
        )
    else:
        split = TTSplit.from_src_and_test(data, test)
        assert split.train.interaction_count + split.test_size == len(df)
        return split
