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

    offsets = np.require(matrix.indptr[:-1], "i4")
    pairs = pa.StructArray.from_arrays([matrix.indices, matrix.data], ["column", "value"])
    lists = pa.ListArray.from_arrays(offsets, pairs)

    users = rng.choice(ml_20m.user_count, 1000, replace=True)
    users = np.require(users, "i4")

    def select():
        res = pc.take(lists, users)

    benchmark(select)


@mark.benchmark()
def test_select_scipy(ml_20m: Dataset, rng: np.random.Generator, benchmark):
    matrix = ml_20m.interactions().matrix().scipy(layout="csr")

    users = rng.choice(ml_20m.user_count, 1000, replace=True)
    users = np.require(users, "i4")

    def select():
        res = matrix[users, :]

    benchmark(select)
