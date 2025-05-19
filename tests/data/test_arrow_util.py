# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pyarrow as pa
import torch

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given

from lenskit.data.arrow import arrow_to_format


@given(nph.arrays(dtype=nph.floating_dtypes(endianness="="), shape=nph.array_shapes(max_dims=1)))
def test_array_to_arrow(data: np.ndarray):
    arr = pa.array(data)

    arr_arrow = arrow_to_format(arr, "arrow")
    assert arr_arrow is arr


@given(nph.arrays(dtype=nph.floating_dtypes(endianness="="), shape=nph.array_shapes(max_dims=1)))
def test_array_to_numpy(data: np.ndarray):
    arr = pa.array(data)

    arr_numpy = arrow_to_format(arr, "numpy")
    assert isinstance(arr_numpy, np.ndarray)
    assert len(arr_numpy) == len(data)


@given(nph.arrays(dtype=nph.floating_dtypes(endianness="="), shape=nph.array_shapes(max_dims=1)))
def test_array_to_torch(data: np.ndarray):
    arr = pa.array(data)

    arr_torch = arrow_to_format(arr, "torch")
    assert isinstance(arr_torch, torch.Tensor)
    assert len(arr_torch) == len(data)
