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
