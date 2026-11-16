# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Test the array hashing code.
"""

import pyarrow as pa

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given

from lenskit._accel import data

SHA1_HEX_LEN = 40


@given(nph.arrays(nph.integer_dtypes(endianness="="), nph.array_shapes(max_dims=1, min_dims=1)))
def test_hash_ints(vec):
    arr = pa.array(vec)
    h = data.hash_array(arr)
    assert isinstance(h, str)
    assert len(h) == SHA1_HEX_LEN


@given(st.lists(st.emails()))
def test_hash_strings(emails):
    arr = pa.array(emails, type=pa.string())
    h = data.hash_array(arr)
    assert isinstance(h, str)
    assert len(h) == SHA1_HEX_LEN


@given(st.lists(st.emails()))
def test_hash_string_views(emails):
    arr = pa.array(emails, type=pa.string_view())
    h = data.hash_array(arr)
    assert isinstance(h, str)
    assert len(h) == SHA1_HEX_LEN
