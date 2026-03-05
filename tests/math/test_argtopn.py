# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given, settings

from lenskit.stats import argtopn


def test_simple_topn():
    positions = argtopn([1.0, 0.0], 1)
    assert len(positions) == 1
    assert positions[0] == 0


def test_simple_topn_rev():
    positions = argtopn([0.0, 1.0], 1)
    assert len(positions) == 1
    assert positions[0] == 1


@given(
    nph.arrays(nph.floating_dtypes(endianness="="), st.integers(0, 5000)), st.integers(min_value=-1)
)
def test_arg_topn(xs, k):
    positions = argtopn(xs, k)
    if k >= 0:
        assert len(positions) <= k
    assert positions.dtype == np.int64
    if k == 0 or np.all(np.isnan(xs)):
        assert len(positions) == 0
        return

    top_xs = xs[positions]

    sort_xs = np.sort(-xs[~np.isnan(xs)])

    # we have the correct number of positions
    if k >= 0:
        assert len(positions) == min(k, np.sum(~np.isnan(xs)))
    else:
        assert len(positions) == np.sum(~np.isnan(xs))
    # all rank positions are valid
    assert np.all(positions >= 0)
    assert np.all(positions < len(xs))
    # all rank positions are unique
    assert len(np.unique(positions)) == len(positions)
    # all ranked items are numbers
    assert not np.any(np.isnan(top_xs))

    # we have the largest values
    if len(positions) < k:
        omitted = np.ones(len(xs), dtype="bool")
        omitted[positions] = False
        if not np.all(np.isnan(xs[omitted])):
            assert np.all(top_xs >= np.nanmax(xs[omitted]))

    # the values are sorted
    if len(top_xs) > 1:
        assert np.all(top_xs[:-1] >= top_xs[1:])

    # the min matches the underlying sort
    if k >= 1:
        assert top_xs[-1] == -sort_xs[min(k - 1, len(sort_xs) - 1)]
    elif np.all(np.isfinite(sort_xs)):
        assert np.all(top_xs == -sort_xs)
