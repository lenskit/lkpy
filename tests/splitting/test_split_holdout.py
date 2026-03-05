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
from lenskit.splitting.holdout import LastFrac, LastN, SampleFrac, SampleN


def test_sample_n(ml_ds: Dataset):
    users = np.random.choice(ml_ds.users.ids(), 5, replace=False)

    s5 = SampleN(5)
    for u in users:
        row = ml_ds.user_row(u)
        assert row is not None
        tst = s5(row)
        mask = np.isin(row.ids(), tst.ids())
        trn = row[~mask]
        assert len(tst) == 5
        assert len(tst) + len(trn) == len(row)

    s10 = SampleN(10)
    for u in users:
        row = ml_ds.user_row(u)
        assert row is not None
        tst = s10(row)
        mask = np.isin(row.ids(), tst.ids())
        trn = row[~mask]
        assert len(tst) == 10
        assert len(tst) + len(trn) == len(row)


def test_sample_frac(ml_ds: Dataset):
    users = np.random.choice(ml_ds.users.ids(), 5, replace=False)

    samp = SampleFrac(0.2)
    for u in users:
        row = ml_ds.user_row(u)
        assert row is not None
        tst = samp(row)
        mask = np.isin(row.ids(), tst.ids())
        trn = row[~mask]
        assert len(tst) + len(trn) == len(row)
        assert len(tst) >= math.floor(len(row) * 0.2)
        assert len(tst) <= math.ceil(len(row) * 0.2)

    samp = SampleFrac(0.5)
    for u in users:
        row = ml_ds.user_row(u)
        assert row is not None
        tst = samp(row)
        mask = np.isin(row.ids(), tst.ids())
        trn = row[~mask]
        assert len(tst) + len(trn) == len(row)
        assert len(tst) >= math.floor(len(row) * 0.5)
        assert len(tst) <= math.ceil(len(row) * 0.5)


def test_last_n(ml_ds: Dataset):
    users = np.random.choice(ml_ds.users.ids(), 5, replace=False)

    samp = LastN(5)
    for u in users:
        row = ml_ds.user_row(u)
        assert row is not None
        tst = samp(row)
        mask = np.isin(row.ids(), tst.ids())
        trn = row[~mask]
        assert len(tst) == 5
        assert len(tst) + len(trn) == len(row)
        assert tst.field("timestamp").min() >= trn.field("timestamp").max()  # type: ignore

    samp = LastN(7)
    for u in users:
        row = ml_ds.user_row(u)
        assert row is not None
        tst = samp(row)
        mask = np.isin(row.ids(), tst.ids())
        trn = row[~mask]
        assert len(tst) == 7
        assert len(tst) + len(trn) == len(row)
        assert tst.field("timestamp").min() >= trn.field("timestamp").max()  # type: ignore


def test_last_frac(ml_ds: Dataset):
    users = np.random.choice(ml_ds.users.ids(), 5, replace=False)

    samp = LastFrac(0.2, "timestamp")
    for u in users:
        row = ml_ds.user_row(u)
        assert row is not None
        tst = samp(row)
        mask = np.isin(row.ids(), tst.ids())
        trn = row[~mask]
        assert len(tst) + len(trn) == len(row)
        assert len(tst) >= math.floor(len(row) * 0.2)
        assert len(tst) <= math.ceil(len(row) * 0.2)
        assert tst.field("timestamp").min() >= trn.field("timestamp").max()  # type: ignore

    samp = LastFrac(0.5, "timestamp")
    for u in users:
        row = ml_ds.user_row(u)
        assert row is not None
        tst = samp(row)
        mask = np.isin(row.ids(), tst.ids())
        trn = row[~mask]
        assert len(tst) + len(trn) == len(row)
        assert len(tst) >= math.floor(len(row) * 0.5)
        assert len(tst) <= math.ceil(len(row) * 0.5)
        assert tst.field("timestamp").min() >= trn.field("timestamp").max()  # type: ignore
