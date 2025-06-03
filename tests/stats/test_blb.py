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


def test_blb_single_array(rng: np.random.Generator):
    "Quick one-array test to fail fast"
    xs = rng.standard_normal(40_000) + 1.0
    mean = np.mean(xs)
    ste = np.std(xs) / 200

    summary = blb_summary(xs, "mean", rng=rng)
    print(summary)
    assert isinstance(summary, dict)
    assert summary["estimate"] == approx(mean)
    assert summary["rep_mean"] == approx(mean, rel=0.05)

    assert summary["ci_lower"] == approx(mean - 1.96 * ste, rel=0.05)
    assert summary["ci_upper"] == approx(mean + 1.96 * ste, rel=0.05)


@mark.slow
@mark.parametrize("size", [1000, 10000, 1000000])
@mark.filterwarnings(r"error:.*ignoring \d+ nonfinite values")
def test_blb_array_normal(rng: np.random.Generator, size: int):
    "Test BLB with arrays of normals."

    TRUE_MEAN = 1.0
    results = []

    # Test: for 1000 runs, do approx. 95% of confidence intervals contain the
    # true mean?

    for i in range(1000):
        xs = rng.normal(1.0, 1.0, size)
        mean = np.mean(xs)

        summary = blb_summary(xs, "mean", rng=rng)
        assert isinstance(summary, dict)
        assert summary["estimate"] == approx(mean)

        results.append(summary)

    n_good = len([r for r in results if r["ci_lower"] <= TRUE_MEAN <= r["ci_upper"]])
    # leave a little wiggle room
    assert n_good >= 925


@mark.skip("need to find better parameters")
@given(
    nph.arrays(shape=st.integers(10000, 1_000_000), dtype=nph.floating_dtypes(endianness="=")),
    st.integers(0),
)
def test_blb_array(xs: NDArray[np.floating], seed: int):
    "Test BLB with more aggressive edge-case hunting."
    xsf = xs[np.isfinite(xs)]
    mean = np.mean(xsf)
    # ignore grotesquely out-of-bounds cases (for now)
    assume(np.isfinite(mean))
    n = len(xsf)
    std = np.std(xsf)
    ste = std / sqrt(n)

    if np.all(np.isfinite(xs)):
        summary = blb_summary(xs, "mean", rng=seed)
    else:
        with warns(DataWarning, match=r"ignoring \d+ nonfinite"):
            summary = blb_summary(xs, "mean", rng=seed)

    assert isinstance(summary, dict)
    assert summary["value"] == approx(mean, nan_ok=True)
    assert summary["mean"] == approx(mean, rel=0.01, nan_ok=True)

    if n == 0:
        assert np.isnan(summary["ci_min"])
        assert np.isnan(summary["ci_max"])
    elif np.allclose(xs, np.min(xs)):
        # standard error is zero
        assert summary["ci_min"] == approx(mean, rel=0.01)
        assert summary["ci_max"] == approx(mean, rel=0.01)
    else:
        assert summary["ci_min"] == approx(mean - 1.96 * ste, rel=0.01)
        assert summary["ci_max"] == approx(mean + 1.96 * ste, rel=0.01)
