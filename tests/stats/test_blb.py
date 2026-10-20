# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os
from concurrent.futures import ThreadPoolExecutor
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
from lenskit.parallel import is_free_threaded
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
    N_TRIALS: ClassVar[int] = 1000

    parameter: float

    def generate_sample(self, size: int, rng: np.random.Generator) -> NDArray[np.float64]: ...
    def compute_stats(
        self, xs: NDArray[np.float64], rng: np.random.Generator
    ) -> dict[str, float]: ...

    def expected_width(self, size: int) -> float | None:
        return None

    @mark.filterwarnings(r"error:.*ignoring \d+ nonfinite values")
    @mark.parametrize("size", [1000, 10000, 50000])
    @mark.skipif(not is_free_threaded(require_active=True), reason="requires free-threading")
    def test_compute(self, size: int, rng: np.random.Generator):
        results = []
        times = []

        rngs = rng.spawn(self.N_TRIALS)
        with ThreadPoolExecutor() as pool:
            for mean, summary, time in pool.map(lambda t: _blb_worker(size, self, t), rngs):
                assert isinstance(summary, dict)
                assert summary["estimate"] == approx(mean)

                results.append(summary)
                times.append(time)

        _log.info("completed %d trials (avg %.2fms / trial)", len(results), np.mean(times) * 1000)
        n_lb_good = len([r for r in results if r["ci_lower"] <= self.parameter])
        f_lb_good = n_lb_good / self.N_TRIALS
        n_ub_good = len([r for r in results if self.parameter <= r["ci_upper"]])
        f_ub_good = n_ub_good / self.N_TRIALS
        n_good = len([r for r in results if r["ci_lower"] <= self.parameter <= r["ci_upper"]])
        f_good = n_good / self.N_TRIALS
        bt = binomtest(n_good, self.N_TRIALS, 0.95)
        _log.info(
            "binomial test for CI hit rate: stat=%.3f, p=%.3g", bt.statistic, bt.pvalue, test=bt
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
        try:
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
        except KeyError:
            pass

        widths = np.array([r["ci_upper"] - r["ci_lower"] for r in results])
        ew = self.expected_width(size)
        _log.info(f"bootstrap CI widths (expected: {ew:.4f}): {describe(widths)}")
        if ew is not None:
            wt = ttest_1samp(widths, ew)
            if wt.pvalue >= 0.05:
                _log.info(
                    "t-test for CI width: stat=%.5f, p=%.3g", wt.statistic, wt.pvalue, test=wt
                )
            else:
                _log.warn(
                    "t-test for CI width: stat=%.5f, p=%.3g", wt.statistic, wt.pvalue, test=wt
                )

            if bt.pvalue >= 0.05:
                _log.info(
                    f"{f_good:.1%} CIs good ({1 - f_lb_good:1%} LB fail, {1 - f_ub_good:.1%} UB fail), p={bt.pvalue:.3g}",
                )
            else:
                _log.error(
                    f"{f_good:.1%} CIs good ({1 - f_lb_good:1%} LB fail, {1 - f_ub_good:.1%} UB fail), p={bt.pvalue:.3g}",
                )

        # leave some wiggle room
        assert bt.pvalue >= 0.05


def _blb_worker(
    size: int, test: CITester, rng: np.random.Generator
) -> tuple[float, dict[str, float], float]:
    xs = test.generate_sample(size, rng)
    mean = np.mean(xs).item()

    timer = Stopwatch()
    s = test.compute_stats(xs, rng)

    return mean, s, timer.elapsed()


class TestParamNormal(CITester):
    """
    Test parametric confidence intervals.

    This test is primarily to ensure that the test machinery itself is working correctly.
    """

    parameter = 1.0
    true_sd = 1.0

    def expected_width(self, size: int):
        se = self.true_sd / np.sqrt(size)
        return 2 * 1.96 * se

    def generate_sample(self, size: int, rng):
        return rng.normal(self.parameter, self.true_sd, size=size)

    def compute_stats(self, xs, rng: np.random.Generator):
        mean = np.mean(xs)
        ssd = np.std(xs, ddof=1)
        sse = ssd / np.sqrt(len(xs))
        return {
            "estimate": mean,
            "ci_lower": mean - 1.96 * sse,
            "ci_upper": mean + 1.96 * sse,
        }


class TestSimpleNormal(CITester):
    parameter = 1.0
    true_sd = 1.0

    def expected_width(self, size: int):
        se = self.true_sd / np.sqrt(size)
        return 2 * 1.96 * se

    def generate_sample(self, size: int, rng):
        return rng.normal(self.parameter, self.true_sd, size=size)

    def compute_stats(self, xs, rng: np.random.Generator):
        return blb_summary(xs, "mean", rng=rng)


class TestSmallBeta(CITester):
    parameter = 2 / 9
    true_sd = np.sqrt(2 * 7 / (9**2 * 10))

    def expected_width(self, size: int):
        se = self.true_sd / np.sqrt(size)
        return 2 * 1.96 * se

    def generate_sample(self, size: int, rng):
        return rng.beta(2, 7, size=size)

    def compute_stats(self, xs, rng: np.random.Generator):
        return blb_summary(xs, "mean", rng=rng)
