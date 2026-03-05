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

from lenskit.data import Dataset
from lenskit.splitting.records import crossfold_records, sample_records


def test_crossfold_records(ml_ds: Dataset):
    splits = crossfold_records(ml_ds, 5)
    splits = list(splits)
    assert len(splits) == 5

    for s in splits:
        # do we have all the data?
        test_count = s.test_size
        assert test_count + s.train.interaction_count == ml_ds.interactions().count()
        test_pairs = set((u, i) for (u, il) in s.test for i in il.ids())
        tdf = s.train.interaction_matrix(format="pandas", field="rating", original_ids=True)
        train_pairs = set(zip(tdf["user_id"], tdf["item_id"]))

        # no overlap
        assert not (test_pairs & train_pairs)
        # union is complete
        assert len(test_pairs | train_pairs) == ml_ds.interactions().count()

    # the test sets are pairwise disjoint
    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue

        p1 = set((u, i) for (u, il) in s1.test for i in il.ids())
        p2 = set((u, i) for (u, il) in s2.test for i in il.ids())
        assert not (p1 & p2)


def test_sample_records_once(ml_ds):
    split = sample_records(ml_ds, size=1000)

    test_count = split.test_size
    assert test_count == 1000
    assert test_count + split.train.interaction_count == ml_ds.interactions().count()
    test_pairs = set((u, i) for (u, il) in split.test for i in il.ids())
    tdf = split.train.interaction_matrix(format="pandas", field="rating", original_ids=True)
    train_pairs = set(zip(tdf["user_id"], tdf["item_id"]))

    # no overlap
    assert not (test_pairs & train_pairs)
    # union is complete
    assert len(test_pairs | train_pairs) == ml_ds.interactions().count()


def test_sample_records(ml_ds):
    splits = sample_records(ml_ds, size=1000, repeats=5)
    splits = list(splits)
    assert len(splits) == 5

    for s in splits:
        test_count = s.test_size
        assert test_count == 1000
        assert test_count + s.train.interaction_count == ml_ds.interactions().count()
        test_pairs = set((u, i) for (u, il) in s.test for i in il.ids())
        tdf = s.train.interaction_matrix(format="pandas", field="rating", original_ids=True)
        train_pairs = set(zip(tdf["user_id"], tdf["item_id"]))

        # no overlap
        assert not (test_pairs & train_pairs)
        # union is complete
        assert len(test_pairs | train_pairs) == ml_ds.interactions().count()

    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue

        p1 = set((u, i) for (u, il) in s1.test for i in il.ids())
        p2 = set((u, i) for (u, il) in s2.test for i in il.ids())
        assert not (p1 & p2)


def test_sample_records_test_only(ml_ds):
    splits = sample_records(ml_ds, size=1000, repeats=1, test_only=True)
    splits = list(splits)
    assert len(splits) == 1

    for s in splits:
        test_count = s.test_size
        assert test_count == 1000
        assert s.train.interaction_count == 0


def test_sample_rows_more_smaller_parts(ml_ds: Dataset):
    splits = sample_records(ml_ds, 500, repeats=10)
    splits = list(splits)
    assert len(splits) == 10

    for s in splits:
        test_count = s.test_size
        assert test_count == 500
        assert test_count + s.train.interaction_count == ml_ds.interactions().count()
        test_pairs = set((u, i) for (u, il) in s.test for i in il.ids())
        tdf = s.train.interaction_matrix(format="pandas", field="rating", original_ids=True)
        train_pairs = set(zip(tdf["user_id"], tdf["item_id"]))

        # no overlap
        assert not (test_pairs & train_pairs)
        # union is complete
        assert len(test_pairs | train_pairs) == ml_ds.interactions().count()

    for s1, s2 in it.product(splits, splits):
        if s1 is s2:
            continue

        p1 = set((u, i) for (u, il) in s1.test for i in il.ids())
        p2 = set((u, i) for (u, il) in s2.test for i in il.ids())
        assert not (p1 & p2)


def test_sample_non_disjoint(ml_ds: Dataset):
    splits = sample_records(ml_ds, 1000, repeats=10, disjoint=False)
    splits = list(splits)
    assert len(splits) == 10

    for s in splits:
        test_count = s.test_size
        assert test_count == 1000
        assert test_count + s.train.interaction_count == ml_ds.interactions().count()
        test_pairs = set((u, i) for (u, il) in s.test for i in il.ids())
        tdf = s.train.interaction_matrix(format="pandas", field="rating", original_ids=True)
        train_pairs = set(zip(tdf["user_id"], tdf["item_id"]))

        # no overlap
        assert not (test_pairs & train_pairs)
        # union is complete
        assert len(test_pairs | train_pairs) == ml_ds.interactions().count()

    # There are enough splits & items we should pick at least one duplicate
    ipairs = (
        (
            set((u, i) for (u, il) in s1.test for i in il.ids()),
            set((u, i) for (u, il) in s2.test for i in il.ids()),
        )
        for (s1, s2) in it.product(splits, splits)
    )
    isizes = [len(i1.intersection(i2)) for (i1, i2) in ipairs]
    assert any(n > 0 for n in isizes)


@pytest.mark.slow
def test_sample_oversize(ml_ds: Dataset):
    splits = sample_records(ml_ds, 10000, repeats=50)
    splits = list(splits)
    assert len(splits) == 50

    for s in splits:
        test_count = s.test_size
        assert test_count + s.train.interaction_count == ml_ds.interactions().count()
        test_pairs = set((u, i) for (u, il) in s.test for i in il.ids())
        tdf = s.train.interaction_matrix(format="pandas", field="rating", original_ids=True)
        train_pairs = set(zip(tdf["user_id"], tdf["item_id"]))

        # no overlap
        assert not (test_pairs & train_pairs)
        # union is complete
        assert len(test_pairs | train_pairs) == ml_ds.interactions().count()
