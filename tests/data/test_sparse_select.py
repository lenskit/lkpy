from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from pytest import mark

from lenskit.data import Dataset
from lenskit.testing import ml_20m


@mark.benchmark()
def test_select_arrow(ml_20m: Dataset, rng: np.random.Generator, benchmark):
    matrix = ml_20m.interactions().matrix().scipy(layout="csr")

    offsets = np.require(matrix.indptr[:-1], np.int32)
    offsets = pa.array(offsets)
    pairs = pa.StructArray.from_arrays(
        [pa.array(matrix.indices), pa.array(matrix.data)], ["column", "value"]
    )
    lists = pa.ListArray.from_arrays(offsets, pairs)

    users = rng.choice(ml_20m.user_count, 1000, replace=True)
    users = np.require(users, "i4")
    users = pa.array(users)

    def select():
        res = lists.take(users)

    benchmark(select)


@mark.benchmark()
def test_select_arrow_sep(ml_20m: Dataset, rng: np.random.Generator, benchmark):
    matrix = ml_20m.interactions().matrix().scipy(layout="csr")

    offsets = np.require(matrix.indptr[:-1], "i4")
    offsets = pa.array(offsets)

    colinds = pa.ListArray.from_arrays(offsets, pa.array(matrix.indices))
    values = pa.ListArray.from_arrays(offsets, pa.array(matrix.data))

    users = rng.choice(ml_20m.user_count, 1000, replace=True)
    users = np.require(users, "i4")
    users = pa.array(users)

    def select():
        _cr = colinds.take(users)
        _vr = values.take(users)

    benchmark(select)


@mark.benchmark()
def test_select_scipy_csr(ml_20m: Dataset, rng: np.random.Generator, benchmark):
    matrix = ml_20m.interactions().matrix().scipy(layout="csr")

    users = rng.choice(ml_20m.user_count, 1000, replace=True)
    users = np.require(users, "i4")

    def select():
        res = matrix[users, :]

    benchmark(select)
