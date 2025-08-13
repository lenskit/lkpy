# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from math import sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import assume, given
from pytest import approx, mark, warns

from lenskit.data.types import NPVector
from lenskit.diagnostics import DataWarning
from lenskit.logging import get_logger
from lenskit.parallel.ray import ensure_cluster, ray_available
from lenskit.random import random_generator
from lenskit.stats import blb_summary

_log = get_logger(__name__)


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
    # assert summary["rep_var"] == approx(ste * ste, rel=0.05)

    assert summary["ci_lower"] == approx(mean - 1.96 * ste, rel=0.01)
    assert summary["ci_upper"] == approx(mean + 1.96 * ste, rel=0.01)


@mark.slow
@mark.skipif(not ray_available(), reason="bulk BLB test requires Ray")
@mark.parametrize("size", [1000, 10_000, 100_000])
@mark.filterwarnings(r"error:.*ignoring \d+ nonfinite values")
def test_blb_array_normal(rng: np.random.Generator, size: int):
    "Test BLB with arrays of normals."
    import ray

    ensure_cluster()
    TRUE_MEAN = 1.0
    TRUE_SD = 1.0
    # TRUE_SVAR = TRUE_SD * TRUE_SD / size
    results = []

    # Test: for NBATCHES * PERBATCH runs, do approx. 95% of confidence intervals
    # contain the true mean?

    worker = ray.remote(num_cpus=2)(_blb_worker)

    NBATCHES = 20
    PERBATCH = 50
    NTRIALS = NBATCHES * PERBATCH
    rngs = rng.spawn(NBATCHES)
    tasks = [worker.remote(PERBATCH, TRUE_MEAN, TRUE_SD, size, t) for t in rngs]
    for task in tasks:
        bres = ray.get(task)
        for mean, summary in bres:
            assert isinstance(summary, dict)
            assert summary["estimate"] == approx(mean)

            results.append(summary)

    n_lb_good = len([r for r in results if r["ci_lower"] <= TRUE_MEAN])
    f_lb_good = n_lb_good / NTRIALS
    n_ub_good = len([r for r in results if TRUE_MEAN <= r["ci_upper"]])
    f_ub_good = n_ub_good / NTRIALS
    n_good = len([r for r in results if r["ci_lower"] <= TRUE_MEAN <= r["ci_upper"]])
    f_good = n_good / NTRIALS
    bt = binomtest(n_good, NTRIALS, 0.95)
    _log.info(
        "{:.1%} CIs good ({:1%} LB fail, {:.1%} UB fail), p={:.3g}".format(
            f_good, 1 - f_lb_good, 1 - f_ub_good, bt.pvalue
        ),
        test=bt,
    )
    # leave some wiggle room
    assert bt.pvalue >= 0.05


def _blb_worker(
    nreps: int, true_mean: float, true_sd: float, size: int, rng: np.random.Generator
) -> list[tuple[float, dict[str, float]]]:
    results = []
    # bf = 0.7 if size > 50_000 else 0.8

    for _i in range(nreps):
        xs = rng.normal(true_mean, true_sd, size)
        mean = np.mean(xs).item()

        results.append(
            (
                mean,
                blb_summary(
                    xs, "mean", rng=rng, b_factor=0.7, s_window=10, r_window=20, rel_tol=0.01
                ),
            )
        )

    return results


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
