# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import torch
from numpy.typing import NDArray

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import assume, given

from lenskit.data.mtarray import MTArray, MTIntArray


@given(
    nph.arrays(
        dtype=st.one_of(nph.integer_dtypes(endianness="="), nph.floating_dtypes(endianness="=")),
        shape=nph.array_shapes(),
    )
)
def test_from_numpy(arr: NDArray[np.generic]):
    # limit to data types that match
    assume(np.all(np.isfinite(arr)))
    mta = MTArray(arr)
    assert mta.shape == arr.shape
    npa = mta.numpy()
    assert npa is arr

    tensor = mta.torch()
    assert tensor.shape == arr.shape
    assert np.all(tensor.numpy() == arr)

    assert np.asarray(mta) is arr


@given(
    nph.arrays(
        dtype=st.one_of(nph.integer_dtypes(endianness="="), nph.floating_dtypes(endianness="=")),
        shape=nph.array_shapes(),
    )
)
def test_from_torch(arr: NDArray[np.generic]):
    # limit to data types that match
    assume(np.all(np.isfinite(arr)))
    ot = torch.from_numpy(arr)
    mta = MTArray(ot)
    assert mta.shape == arr.shape
    tensor = mta.torch()
    assert tensor is ot

    npa = mta.numpy()
    assert npa.shape == arr.shape
    assert np.all(npa == arr)


@given(st.lists(st.integers(min_value=np.iinfo(np.int64).min, max_value=np.iinfo(np.int64).max)))
def test_from_list(xs: list[int]):
    # limit to data types that match
    mta = MTIntArray(xs)
    assert mta.shape == (len(xs),)

    npa = mta.numpy()
    assert np.all(npa == xs)

    tensor = mta.torch()
    assert np.all(tensor.numpy() == xs)
