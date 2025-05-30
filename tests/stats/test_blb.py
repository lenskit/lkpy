# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given


@given(nph.arrays(shape=st.integers(10000, 1_000_000), dtype=nph.floating_dtypes(endianness="=")))
def test_blb_array():
    pass
