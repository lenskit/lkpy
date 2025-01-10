import numpy as np

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given, settings

from lenskit.stats import argtopk


def test_simple_topk():
    positions = argtopk([1.0, 0.0], 1)
    assert len(positions) == 1
    assert positions[0] == 0


def test_simple_topk_rev():
    positions = argtopk([0.0, 1.0], 1)
    assert len(positions) == 1
    assert positions[0] == 1


@given(
    nph.arrays(nph.floating_dtypes(endianness="="), st.integers(0, 5000)), st.integers(min_value=-1)
)
def test_arg_topk(xs, k):
    positions = argtopk(xs, k)
    if k >= 0:
        assert len(positions) <= k
    assert positions.dtype == np.int64
    if k == 0 or np.all(np.isnan(xs)):
        assert len(positions) == 0
        return

    top_xs = xs[positions]

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
        omitted = np.ones(len(xs), dtype=np.bool)
        omitted[positions] = False
        if not np.all(np.isnan(xs[omitted])):
            assert np.all(top_xs >= np.nanmax(xs[omitted]))

    # the values are sorted
    if len(top_xs) > 1:
        assert np.all(top_xs[:-1] >= top_xs[1:])
