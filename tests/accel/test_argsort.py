# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pyarrow as pa

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given

from lenskit._accel import data


@given(
    nph.arrays(
        nph.floating_dtypes(endianness="="),
        nph.array_shapes(max_dims=1),
        elements={"allow_nan": False, "allow_infinity": False},
    )
)
def test_sort_floats(arr):
    a2 = pa.array(arr)

    idx = data.argsort_descending(a2)
    assert len(idx) == len(arr)
    assert set(idx.to_pylist()) == set(range(len(arr)))

    items = arr[idx]

    # descending order
    for i in range(1, len(arr)):
        assert items[i] <= items[i - 1]


@given(
    nph.arrays(
        nph.integer_dtypes(endianness="="),
        nph.array_shapes(max_dims=1),
        elements={"allow_nan": False, "allow_infinity": False},
    )
)
def test_sort_ints(arr):
    a2 = pa.array(arr)

    idx = data.argsort_descending(a2)
    assert len(idx) == len(arr)
    assert set(idx.to_pylist()) == set(range(len(arr)))

    items = arr[idx]

    # descending order
    for i in range(1, len(arr)):
        assert items[i] <= items[i - 1]
