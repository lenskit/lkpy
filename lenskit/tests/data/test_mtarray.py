# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pyarrow as pa
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


@given(
    nph.arrays(
        dtype=st.one_of(nph.integer_dtypes(endianness="="), nph.floating_dtypes(endianness="=")),
        shape=nph.array_shapes(),
    )
)
def test_from_arrow(arr: NDArray[np.generic]):
    # limit to data types that match
    assume(np.all(np.isfinite(arr)))
    if len(arr.shape) == 1:
        oarr = pa.array(arr)
    else:
        oarr = pa.Tensor.from_numpy(arr)
    mta = MTArray(oarr)
    assert mta.shape == arr.shape

    a2 = mta.arrow()
    assert a2 is oarr

    npa = mta.numpy()
    assert npa.shape == arr.shape
    assert np.all(npa == arr)


@given(
    nph.arrays(
        dtype=st.one_of(nph.integer_dtypes(endianness="="), nph.floating_dtypes(endianness="=")),
        shape=nph.array_shapes(),
    ),
    st.sampled_from(["numpy", "arrow", "torch"]),
    st.sampled_from(["numpy", "arrow", "torch"]),
)
def test_data_combo(arr: NDArray[np.generic], src, tgt):
    # limit to data types that match
    assume(np.all(np.isfinite(arr)))

    match src:
        case "numpy":
            mta = MTArray(arr)
        case "torch":
            mta = MTArray(torch.from_numpy(arr))
        case "arrow" if len(arr.shape) == 1:
            a2 = pa.array(arr)
            assert isinstance(a2, pa.Array)
            mta = MTArray(a2)
        case "arrow":
            mta = MTArray(pa.Tensor.from_numpy(arr))

    assert not isinstance(mta._convertible(), pa.Scalar)

    assert mta.shape == arr.shape

    try:
        match tgt:
            case "numpy":
                out = mta.numpy()
            case "arrow":
                out = mta.arrow()
            case "torch":
                out = mta.torch()
    except Exception as e:
        print("internal:", type(mta._unknown), mta._unknown)
        raise e

    if isinstance(out, pa.Array):
        assert len(arr.shape) == 1
        assert len(out) == arr.shape[0]
    else:
        assert out.shape == arr.shape
    assert np.all(np.asarray(out) == arr)


@given(st.lists(st.integers(min_value=np.iinfo(np.int64).min, max_value=np.iinfo(np.int64).max)))
def test_from_list(xs: list[int]):
    # limit to data types that match
    mta = MTIntArray(xs)
    assert mta.shape == (len(xs),)

    npa = mta.numpy()
    assert np.all(npa == xs)

    tensor = mta.torch()
    assert np.all(tensor.numpy() == xs)
