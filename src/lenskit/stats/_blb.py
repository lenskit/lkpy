# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, TypeAlias, TypeVar

import numpy as np
import pandas as pd
import scipy.stats
from numpy.typing import NDArray

from lenskit.diagnostics import DataWarning
from lenskit.logging import Tracer, get_logger, get_tracer
from lenskit.random import RNGInput, random_generator

from ._distributions import ci_quantiles

F = TypeVar("F", bound=np.floating, covariant=True)

SummaryStat: TypeAlias = Literal["mean"]

_log = get_logger(__name__)
STD_NORM = scipy.stats.norm()

# dummy assignment to typecheck that we have correctly typed weighted average
__dummy_avg: WeightedStatistic = np.average


class WeightedStatistic(Protocol):
    """
    Callable interface for weighted statistics, required by the Bag of Little Bootstraps.
    """

    def __call__(
        self,
        a: NDArray[np.floating[Any]],
        /,
        *,
        weights: NDArray[np.floating[Any] | np.integer[Any]] | None = None,
        axis: int | None = None,
    ) -> np.floating[Any]: ...


def blb_summary(
    xs: NDArray[F],
    stat: SummaryStat,
    *,
    ci_width: float = 0.95,
    b_factor: float = 0.7,
    rel_tol: float = 0.02,
    s_window: int = 10,
    r_window: int = 50,
    rng: RNGInput = None,
) -> dict[str, float]:
    r"""
    Summarize one or more statistics using the Bag of Little Bootstraps
    :cite:p:`blb`.

    This is a direct, sequential implementation of Bag of Little Bootstraps as
    described in the original paper :cite:p:`blb`, with automatic
    convergence-based termination.

    Args:
        xs:
            The array of values to summarize.
        stat:
            The statistic to compute.  The Bag of Little Bootstraps requires
            statistics to support weighted computation (this is what allows it
            to speed up the bootstrap procedure).
        ci_width:
            The width of the confidence interval to estimate.
        b_factor:
            The shrinking factor :math:`\gamma` to use to derive subsample
            sizes. Each subsample has size :math:`N^{\gamma}`.
        rel_tol:
            The relative tolerance for detecting convergence.
        s_window:
            The window length for detecting convergence in the outer subset loop
            (and minimum number of subsets).
        r_window:
            The window length for detecting convergence in the inner replication
            loop (and minimum number of replicates per subset).
        rng:
            The RNG or seed for randomization.

    Returns:
        A dictionary of statistical results of the statistic.
    """
    if stat != "mean":
        raise ValueError(f"unsupported statistic {stat}")

    n = len(xs)
    mask = np.isfinite(xs)
    nfinite = np.sum(mask)
    if nfinite < n:
        warnings.warn(f"ignoring {n - nfinite} nonfinite values", DataWarning, stacklevel=2)

    xs = xs[mask]
    est = np.average(xs).item()

    rng = random_generator(rng)
    config = _BLBConfig(
        statistic=np.average,
        ci_width=ci_width,
        rel_tol=rel_tol,
        s_window=s_window,
        r_window=r_window,
        b_factor=b_factor,
    )
    bootstrapper = _BLBootstrapper(config, rng)

    result = bootstrapper.run_bootstraps(xs)

    result = {
        "estimate": est,
        "rep_mean": result.rep_mean,
        "rep_var": result.rep_var,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
    }

    return result


@dataclass
class _BootResult:
    estimate: float
    "Statistic computed on original data."

    rep_mean: float
    "Mean of the statistic computed on the replicates."
    rep_var: float
    "Variance of the statistic computed on the replicates."
    ci_lower: float
    "CI lower bound."
    ci_upper: float
    "CI upper bound."
    samples: pd.DataFrame | None = None
    "Raw sample data."


@dataclass
class _BLBConfig:
    statistic: WeightedStatistic
    ci_width: float
    rel_tol: float
    s_window: int
    r_window: int
    b_factor: float

    @property
    def ci_margin(self) -> float:
        return 0.5 * (1 - self.ci_width)


class _BLBootstrapper:
    """
    Implementation of BLB computation.
    """

    _tracer: Tracer
    config: _BLBConfig
    _ci_qmin: float
    _ci_qmax: float

    rng: np.random.Generator

    def __init__(self, config, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.ss_stats = {}

        self._tracer = get_tracer(_log, stat=config.statistic.__name__)  # type: ignore

    def run_bootstraps(self, xs: NDArray[F]) -> _BootResult:
        n = len(xs)
        self._ci_qmin, self._ci_qmax = ci_quantiles(self.config.ci_width, expand=n)
        b = int(n**self.config.b_factor)

        self._tracer.add_bindings(n=n, b=b)
        _log.debug("starting bootstrap", stat=self.config.statistic.__name__, n=len(xs))  # type: ignore
        ss_frames = {}

        estimate = float(self.config.statistic(xs))

        means = StatAccum(np.mean)
        vars = StatAccum(np.mean)
        lbs = StatAccum(np.mean)
        ubs = StatAccum(np.mean)

        self._tracer.trace("let's go!")

        for i, ss in enumerate(self.blb_subsets(n, b)):
            self._tracer.add_bindings(subset=i)
            self._tracer.trace("starting subset")
            res = self.measure_subset(xs, ss, estimate)
            ss_frames[i] = res.samples
            means.record(res.rep_mean)
            vars.record(res.rep_var)
            lbs.record(res.ci_lower)
            ubs.record(res.ci_upper)
            if self._check_convergence(
                means, vars, lbs, ubs, tol=self.config.rel_tol, w=self.config.s_window
            ):
                break

        return _BootResult(
            estimate,
            means.statistic,
            vars.statistic,
            lbs.statistic,
            ubs.statistic,
            pd.concat(ss_frames, names=["subset"]),
        )

    def blb_subsets(self, n: int, b: int):
        while True:
            yield self.rng.choice(n, b, replace=False)

    def measure_subset(self, xs: NDArray[F], ss: NDArray[np.int64], estimate: float) -> _BootResult:
        b = len(ss)
        n = len(xs)
        xss = xs[ss]

        means = StatAccum(np.mean)
        vars = StatAccum(np.var)
        lbs = StatAccum(None)
        ubs = StatAccum(None)

        for i, weights in enumerate(self.miniboot_weights(n, b)):
            self._tracer.add_bindings(rep=i)
            self._tracer.trace("starting replicate")
            assert weights.shape == (b,)
            assert np.sum(weights) == n
            stat = self.config.statistic(xss, weights=weights)
            means.record(stat)
            vars.record(stat)

            stats = means.values
            # ql, qh = _bca_range(estimate, stats, self.config.ci_margin, accel)
            # self._tracer.trace("bias-corrected quantiles: [%.4f, %.4f]", ql, qh, accel=accel)
            lb, ub = np.quantile(stats, [self._ci_qmin, self._ci_qmax])
            # lb, ub = np.quantile(stats, [ql, qh])
            self._tracer.trace("CI bounds: %f < s < %f", lb, ub)
            lbs.record(stat, lb)
            ubs.record(stat, ub)
            del stats

            if self._check_convergence(
                means, vars, lbs, ubs, tol=self.config.rel_tol, w=self.config.r_window
            ):
                break

        df = pd.DataFrame({"statistic": means.values})
        df.index.name = "iter"
        self._tracer.remove_bindings("rep")
        return _BootResult(
            estimate, means.statistic, vars.statistic, lbs.statistic, ubs.statistic, df
        )

    def miniboot_weights(self, n: int, b: int):
        flat = np.full(b, 1.0 / b)

        while True:
            yield self.rng.multinomial(n, flat)

    def _check_convergence(self, *arrays: StatAccum, tol: float, w: int) -> bool:
        gaps = np.zeros(w)
        for arr in arrays:
            if len(arr) < w + 1:
                return False

            stats = arr.stat_history
            cur = arr.statistic
            gaps += np.abs(stats[-(w + 1) : -1] - cur) / np.abs(cur)

        gaps /= len(arrays)
        self._tracer.trace("max gap: %.3f", np.max(gaps))
        return np.all(gaps < tol).item()


class StatAccum:
    INIT_SIZE: ClassVar[int] = 100

    _stat_func: Callable[[NDArray[np.floating[Any]]], np.floating[Any]]

    _len: int = 0
    _values: NDArray[np.float64]
    _cum_stat: NDArray[np.float64]

    def __init__(self, stat: Callable[[NDArray[np.floating[Any]]], np.floating[Any]]):
        self._stat_func = stat

        self._values = np.zeros(self.INIT_SIZE)
        self._cum_stat = np.zeros(self.INIT_SIZE)

    @property
    def values(self) -> NDArray[np.float64]:
        return self._values[: self._len]

    @property
    def statistic(self) -> float:
        if self._len:
            return self._cum_stat[self._len - 1]
        else:
            return np.nan

    @property
    def stat_history(self) -> NDArray[np.float64]:
        return self._cum_stat[: self._len]

    def record(
        self, x: float | np.floating[Any], stat: float | np.floating[Any] | None = None
    ) -> None:
        "Record a new value in the accumulator."
        self._expand_if_needed()
        i = self._len
        self._len += 1

        # record and update the cumulative mean
        self._values[i] = x
        if stat is None:
            stat = self._stat_func(self.values)
        self._cum_stat[i] = stat

    def _expand_if_needed(self):
        cap = len(self._values)
        if cap == self._len:
            self._values.resize(cap * 2)
            self._cum_stat.resize(cap * 2)

    def __len__(self):
        return self._len
