# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

from pytest import fixture, mark, skip

from lenskit._accel import data as _data_accel
from lenskit.data import Dataset

ML_32M = Path("data/ml-32m")


@fixture(scope="module")
def ml_32m():
    if not ML_32M.exists():
        skip("ML32M not available")

    yield Dataset.load(ML_32M)


@mark.benchmark(group="coordinates", max_time=5)
def test_rc_set(ml_32m: Dataset, rng: np.random.Generator, benchmark):
    users = rng.choice(ml_32m.user_count, size=1000)
    items = rng.choice(ml_32m.item_count, size=1000)

    mat = ml_32m.interactions("rating").matrix()
    rcs = mat._rc_set

    def search():
        for u, i in zip(users, items):
            rcs.contains_pair(u, i)

    benchmark(search)


@mark.benchmark(group="coordinates", max_time=5)
def test_coord_table(ml_32m: Dataset, rng: np.random.Generator, benchmark):
    users = rng.choice(ml_32m.user_count, size=1000)
    items = rng.choice(ml_32m.item_count, size=1000)

    mat = ml_32m.interactions("rating").matrix()
    coo = _data_accel.CoordinateTable(2)

    tbl = mat.arrow(attributes=[])
    coo.extend(tbl.to_batches())

    def search():
        for u, i in zip(users, items):
            coo.contains_pair(u, i)

    benchmark(search)
