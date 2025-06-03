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
from numpy.typing import NDArray

from lenskit.diagnostics import DataWarning
from lenskit.random import RNGInput, random_generator

F = TypeVar("F", bound=np.floating, covariant=True)

SummaryStat: TypeAlias = Literal["mean"]

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
    ) -> np.floating[Any]: ...


def blb_summary(
    xs: NDArray[F],
    stat: SummaryStat,
    *,
    ci_width: float = 0.95,
    b_factor: float = 0.8,
    rel_tol: float = 0.05,
    s_window: int = 3,
    r_window: int = 20,
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
            The width of the confidence interval to estimat.e
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

    mask = np.isfinite(xs)
    if ninf := int(np.sum(~mask)):
        warnings.warn(f"ignoring {ninf} nonfinite values", DataWarning, stacklevel=2)

    xs = xs[mask]
    est = np.average(xs).item()

    rng = random_generator(rng)
    bootstrapper = _BLBootstrapper(np.average, ci_width, rel_tol, s_window, r_window, b_factor, rng)

    result = bootstrapper.run_bootstraps(xs)

    result = {
        "estimate": est,
        "rep_mean": result.mean,
        "rep_var": result.rep_var,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
    }

    return result


@dataclass
class _BootResult:
    mean: float
    rep_var: float
    ci_lower: float
    ci_upper: float
    samples: pd.DataFrame


class _BLBootstrapper:
    """
    Implementation of BLB computation.
    """

    statistic: WeightedStatistic
    ci_width: float
    _ci_qmin: float
    _ci_qmax: float

    tolerance: float
    s_window: int
    r_window: int
    b_factor: float
    rng: np.random.Generator

    def __init__(
        self,
        stat: WeightedStatistic,
        ci_width: float,
        tol: float,
        s_w: int,
        r_w: int,
        b_factor: float,
        rng: np.random.Generator,
    ):
        self.statistic = stat
        self.ci_width = ci_width
        self.tolerance = tol
        self.s_window = s_w
        self.r_window = r_w
        self.b_factor = b_factor
        self.rng = rng
        self.ss_stats = {}

        alpha = 1 - ci_width
        self._ci_qmin = 0.5 * alpha
        self._ci_qmax = 1 - 0.5 * alpha

    def run_bootstraps(self, xs: NDArray[F]) -> _BootResult:
        ss_frames = {}

        means = StatAccum(np.mean)
        vars = StatAccum(np.mean)
        lbs = StatAccum(np.mean)
        ubs = StatAccum(np.mean)

        for i, ss in enumerate(self.blb_subsets(xs)):
            res = self.measure_subset(xs, ss)
            ss_frames[i] = res.samples
            means.record(res.mean)
            lbs.record(res.ci_lower)
            ubs.record(res.ci_upper)
            if _check_convergence(means, vars, lbs, ubs, tol=self.tolerance, w=self.s_window):
                break

        return _BootResult(
            means.statistic,
            vars.statistic,
            lbs.statistic,
            ubs.statistic,
            pd.concat(ss_frames, names=["subset"]),
        )

    def blb_subsets(self, xs: NDArray[F]):
        b = int(len(xs) ** self.b_factor)

        while True:
            yield self.rng.choice(len(xs), b, replace=False)

    def measure_subset(self, xs: NDArray[F], ss: NDArray[np.int64]) -> _BootResult:
        b = len(ss)
        n = len(xs)
        xss = xs[ss]

        values = []
        means = StatAccum(np.mean)
        svs = StatAccum(np.var)
        lbs = StatAccum(lambda a: np.quantile(a, self._ci_qmin))
        ubs = StatAccum(lambda a: np.quantile(a, self._ci_qmax))

        for weights in self.miniboot_weights(n, b):
            assert weights.shape == (b,)
            assert np.sum(weights) == n
            stat = self.statistic(xss, weights=weights)
            values.append(stat)
            means.record(stat)
            lbs.record(stat)
            ubs.record(stat)

            if _check_convergence(means, svs, lbs, ubs, tol=self.tolerance, w=self.r_window):
                break

        df = pd.DataFrame({"statistic": values})
        df.index.name = "iter"
        return _BootResult(means.statistic, svs.statistic, lbs.statistic, ubs.statistic, df)

    def miniboot_weights(self, n: int, b: int):
        flat = np.full(b, 1.0 / b)

        while True:
            yield self.rng.multinomial(n, flat)


def _check_convergence(*arrays: StatAccum, tol: float, w: int) -> bool:
    gaps = np.zeros(w)
    for arr in arrays:
        if len(arr) < w + 1:
            return False
        stats = arr.stat_history
        cur = stats[-1]
        gaps += np.abs(stats[-(w + 1) : -1] - cur) / np.abs(cur)

    gaps /= len(arrays)
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
            return self._cum_stat[-1]
        else:
            return np.nan

    @property
    def stat_history(self) -> NDArray[np.float64]:
        return self._cum_stat[: self._len]

    def record(self, x: float | np.floating[Any]) -> None:
        "Record a new value in the accumulator."
        self._expand_if_needed()
        i = self._len
        self._len += 1

        # record and update the cumulative mean
        self._values[i] = x
        self._cum_stat[i] = self._stat_func(self.values)

    def _expand_if_needed(self):
        cap = len(self._values)
        if cap == self._len:
            self._values.resize(cap * 2)
            self._cum_stat.resize(cap * 2)

    def __len__(self):
        return self._len
