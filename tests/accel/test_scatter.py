# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import Any

import numpy as np
import pyarrow as pa

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given

from lenskit._accel import data


@given(
    st.data(),
    nph.arrays(
        st.one_of(
            nph.integer_dtypes(endianness="="),
            nph.floating_dtypes(endianness="=", sizes=(16, 32, 64)),
        ),
        nph.array_shapes(max_dims=1),
    ),
    st.sampled_from([np.int32, np.int64]),
)
def test_scatter_dst_array(hd: st.DataObject, dst: np.ndarray[tuple[int], Any], idx_t: np.dtype):
    size = len(dst)
    idx = np.asarray(
        list(hd.draw(st.sets(st.integers(min_value=0, max_value=size - 1)))), dtype=idx_t
    )
    src = hd.draw(nph.arrays(dst.dtype, len(idx)))

    dst_a = pa.array(dst)
    idx_a = pa.array(idx)
    src_a = pa.array(src)

    arr_a = data.scatter_array(dst_a, idx_a, src_a)
    assert isinstance(arr_a, pa.Array)
    assert arr_a.type == dst_a.type
    assert arr_a.null_count == 0

    arr = arr_a.to_numpy()

    assert np.array_equal(arr[idx], src, equal_nan=True)


@given(
    st.data(),
    st.integers(0, 16 * 1024 + 1),
    st.sampled_from([np.int32, np.int64]),
)
def test_scatter_dst_size(hd: st.DataObject, size, idx_t: np.dtype):
    if size:
        idx = np.asarray(
            list(hd.draw(st.sets(st.integers(min_value=0, max_value=size - 1)))), dtype=idx_t
        )
    else:
        idx = np.asarray([], dtype=idx_t)
    src = hd.draw(
        nph.arrays(
            st.one_of(
                nph.integer_dtypes(endianness="="),
                nph.floating_dtypes(endianness="=", sizes=(16, 32, 64)),
            ),
            len(idx),
        )
    )

    idx_a = pa.array(idx)
    src_a = pa.array(src)

    arr_a = data.scatter_array_empty(size, idx_a, src_a)
    assert isinstance(arr_a, pa.Array)
    assert arr_a.type == src_a.type
    assert arr_a.null_count == size - len(src)

    arr = arr_a.to_numpy(zero_copy_only=False)

    assert np.array_equal(arr[idx], src, equal_nan=True)
