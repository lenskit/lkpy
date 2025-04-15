# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pyarrow as pa
import scipy.sparse as sps
import torch

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import HealthCheck, given, note, settings
from pytest import approx, mark, skip

from lenskit.torch import safe_tensor


@given(
    nph.arrays(
        st.one_of(nph.floating_dtypes(endianness="="), nph.integer_dtypes(endianness="=")),
        nph.array_shapes(max_dims=3),
    )
)
def test_safe_tensor_from_numpy(arr):
    t = safe_tensor(arr)
    assert torch.is_tensor(t)

    assert t.shape == arr.shape


@given(
    nph.arrays(
        st.one_of(nph.floating_dtypes(endianness="="), nph.integer_dtypes(endianness="=")),
        nph.array_shapes(max_dims=1),
    )
)
def test_safe_tensor_from_arrow(arr):
    arrow = pa.array(arr)
    t = safe_tensor(arrow)
    assert torch.is_tensor(t)

    assert t.shape == arr.shape
