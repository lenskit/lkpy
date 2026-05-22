# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import hypothesis.strategies as st
from hypothesis import assume, given

from lenskit.data.types import Extent


@given(st.integers(0), st.integers(0))
def test_extent_size(lb: int, ub: int):
    assume(ub >= lb)

    extent = Extent(lb, ub)
    assert extent.start == lb
    assert extent.end == ub
    assert extent.size == ub - lb
