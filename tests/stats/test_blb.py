# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os
from math import sqrt
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest, describe, ttest_1samp, ttest_rel

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import assume, given
from pytest import approx, mark, warns

from lenskit.data.types import NPVector
from lenskit.diagnostics import DataWarning
from lenskit.logging import Stopwatch, get_logger
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


class CITester:
    NBATCHES: ClassVar[int] = 20
    PERBATCH: ClassVar[int] = int(os.environ.get("BLB_TRIALS_PER_BATCH", 50))

    parameter: float
    expected_width: float

    def generate_sample(self, rng: np.random.Generator) -> NDArray[np.float64]: ...

    @mark.filterwarnings(r"error:.*ignoring \d+ nonfinite values")
    @mark.parametrize("size", [1000])
    def test_compute(self, size: int, rng: np.random.Generator):
        import ray

        ensure_cluster()

        results = []
        times = []
        n_trials = self.NBATCHES * self.PERBATCH

        worker = ray.remote(num_cpus=2)(_blb_worker)
        rngs = rng.spawn(self.NBATCHES)
        tasks = [worker.remote(self.PERBATCH, size, self, t) for t in rngs]
        for task in tasks:
            bres = ray.get(task)
            for mean, summary, time in bres:
                assert isinstance(summary, dict)
                assert summary["estimate"] == approx(mean)

                results.append(summary)
                times.append(time)

        _log.info("completed %d trials (avg %.2fms / trial)", len(results), np.mean(times) * 1000)
        n_lb_good = len([r for r in results if r["ci_lower"] <= self.parameter])
        f_lb_good = n_lb_good / n_trials
        n_ub_good = len([r for r in results if self.parameter <= r["ci_upper"]])
        f_ub_good = n_ub_good / n_trials
        n_good = len([r for r in results if r["ci_lower"] <= self.parameter <= r["ci_upper"]])
        f_good = n_good / n_trials
        bt = binomtest(n_good, n_trials, 0.95)
        _log.info(
            "binomal test for CI hit rate: stat=%.3f, p=%.3g", bt.statistic, bt.pvalue, test=bt
        )

        smeans = np.array([r["estimate"] for r in results])
        smt = ttest_1samp(smeans, self.parameter)
        _log.info("sample means: %s", describe(smeans))
        if smt.pvalue >= 0.05:
            _log.info(
                "t-test for sample means: stat=%.5f, p=%.3g", smt.statistic, smt.pvalue, test=smt
            )
        else:
            _log.warn(
                "t-test for sample means: stat=%.5f, p=%.3g", smt.statistic, smt.pvalue, test=smt
            )
        rmeans = np.array([r["rep_mean"] for r in results])
        rmt = ttest_rel(rmeans, smeans)
        _log.info("bootstrap means: %s", describe(rmeans))
        if rmt.pvalue >= 0.05:
            _log.info(
                "t-test for CI centers: stat=%.5f, p=%.3g", rmt.statistic, rmt.pvalue, test=rmt
            )
        else:
            _log.warn(
                "t-test for CI centers: stat=%.5f, p=%.3g", rmt.statistic, rmt.pvalue, test=rmt
            )

        widths = np.array([r["ci_upper"] - r["ci_lower"] for r in results])
        _log.info(
            "bootstrap CI widths (expected: {:.4f}): {}".format(
                self.expected_width, describe(widths)
            )
        )
        wt = ttest_1samp(widths, self.expected_width)
        if wt.pvalue >= 0.05:
            _log.info("t-test for CI width: stat=%.5f, p=%.3g", wt.statistic, wt.pvalue, test=wt)
        else:
            _log.warn("t-test for CI width: stat=%.5f, p=%.3g", wt.statistic, wt.pvalue, test=wt)

        if bt.pvalue >= 0.05:
            _log.info(
                "{:.1%} CIs good ({:1%} LB fail, {:.1%} UB fail), p={:.3g}".format(
                    f_good, 1 - f_lb_good, 1 - f_ub_good, bt.pvalue
                ),
            )
        else:
            _log.error(
                "{:.1%} CIs good ({:1%} LB fail, {:.1%} UB fail), p={:.3g}".format(
                    f_good, 1 - f_lb_good, 1 - f_ub_good, bt.pvalue
                ),
            )

        # leave some wiggle room
        assert bt.pvalue >= 0.05


def _blb_worker(
    nreps: int, size: int, test: CITester, rng: np.random.Generator
) -> list[tuple[float, dict[str, float], float]]:
    results = []

    for _i in range(nreps):
        xs = test.generate_sample(size, rng)
        mean = np.mean(xs).item()

        timer = Stopwatch()
        s = blb_summary(xs, "mean", rng=rng, b_factor=0.8, s_window=20, r_window=50, rel_tol=0.01)

        results.append((mean, s, timer.elapsed()))

    return results


class TestSimpleNormal(CITester):
    parameter = 1.0
    true_sd = 1.0

    def generate_sample(self, size: int, rng):
        return rng.normal(self.parameter, self.true_sd, size=size)
