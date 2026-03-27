# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pyarrow as pa

from pytest import fixture, mark, skip

from lenskit.data import Dataset

ML_32M = Path("data/ml-32m")


@fixture(scope="module")
def ml_32m():
    if not ML_32M.exists():
        skip("ML32M not available")

    yield Dataset.load(ML_32M)


@mark.benchmark()
def test_neg_unverified(rng: np.random.Generator, ml_ds: Dataset, benchmark):
    matrix = ml_ds.interactions().matrix()

    users = rng.choice(ml_ds.user_count, 8192, replace=True)
    users = np.require(users, "i4")

    def sample():
        _items = matrix.sample_negatives(users, verify=False, rng=rng)

    benchmark(sample)


@mark.benchmark()
def test_neg_verified(rng: np.random.Generator, ml_ds: Dataset, benchmark):
    matrix = ml_ds.interactions().matrix()

    users = rng.choice(ml_ds.user_count, 8192, replace=True)
    users = np.require(users, "i4")

    def sample():
        _items = matrix.sample_negatives(users, rng=rng)

    benchmark(sample)


@mark.benchmark()
def test_neg_vweighted(rng: np.random.Generator, ml_ds: Dataset, benchmark):
    matrix = ml_ds.interactions().matrix()

    users = rng.choice(ml_ds.user_count, 8192, replace=True)
    users = np.require(users, "i4")

    def sample():
        _items = matrix.sample_negatives(users, weighting="popularity", rng=rng)

    benchmark(sample)
