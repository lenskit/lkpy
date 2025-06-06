# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Test confidence interval utilities.
"""

import hypothesis.strategies as st
from hypothesis import given
from pytest import approx

from lenskit.stats._distributions import ci_quantiles


@given(st.floats(0, 1, exclude_max=True, exclude_min=True))
def test_ci_bounds(width: float):
    qlo, qhi = ci_quantiles(width)
    assert qhi - qlo == approx(width)
    assert 1 - qhi == approx(qlo)
