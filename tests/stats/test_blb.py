# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from math import sqrt

import numpy as np
from numpy.typing import NDArray

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import assume, given
from pytest import approx, mark, warns

from lenskit.data.types import NPVector
from lenskit.diagnostics import DataWarning
from lenskit.random import random_generator
from lenskit.stats import blb_summary


@given(
    st.integers(1000, 1_000_000),
    nph.floating_dtypes(endianness="="),
    st.integers(10, 20),
    st.integers(100, 300),
    st.integers(0),
)
@mark.filterwarnings(r"error:.*ignoring \d+ nonfinite values")
def test_blb_array_normal(n, dtype, s: int, r: int, seed):
    "Test BLB with arrays of standard normals."
    rng = random_generator(seed)
    xs = rng.standard_normal(n).astype(dtype)
    mean = np.mean(xs)
    n = len(xs)
    std = np.std(xs)
    ste = std / sqrt(n)

    summary = blb_summary(xs, "mean", s=s, r=r, rng=rng)
    assert isinstance(summary, dict)
    assert summary["value"] == approx(mean)
    assert summary["mean"] == approx(mean, rel=0.05)

    assert summary["low"] == approx(mean - 1.96 * ste, rel=0.05)
    assert summary["high"] == approx(mean + 1.96 * ste, rel=0.05)


@given(
    nph.arrays(shape=st.integers(10000, 1_000_000), dtype=nph.floating_dtypes(endianness="=")),
    st.integers(10, 20),
    st.integers(100, 300),
    st.integers(0),
)
def test_blb_array(xs: NDArray[np.floating], s: int, r: int, seed: int):
    "Test BLB with more aggressive edge-case hunting."
    xsf = xs[np.isfinite(xs)]
    mean = np.mean(xsf)
    # ignore grotesquely out-of-bounds cases (for now)
    assume(np.isfinite(mean))
    n = len(xsf)
    std = np.std(xsf)
    ste = std / sqrt(n)

    if np.all(np.isfinite(xs)):
        summary = blb_summary(xs, "mean", s=s, r=r, rng=seed)
    else:
        with warns(DataWarning, match=r"ignoring \d+ nonfinite"):
            summary = blb_summary(xs, "mean", s=s, r=r, rng=seed)

    assert isinstance(summary, dict)
    assert summary["value"] == approx(mean, nan_ok=True)
    assert summary["mean"] == approx(mean, rel=0.01, nan_ok=True)

    if n == 0:
        assert np.isnan(summary["low"])
        assert np.isnan(summary["high"])
    elif np.allclose(xs, np.min(xs)):
        # standard error is zero
        assert summary["low"] == approx(mean, rel=0.01)
        assert summary["high"] == approx(mean, rel=0.01)
    else:
        assert summary["low"] == approx(mean - 1.96 * ste, rel=0.01)
        assert summary["high"] == approx(mean + 1.96 * ste, rel=0.01)
