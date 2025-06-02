# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import warnings
from typing import Any, ClassVar, Literal, Protocol, TypeAlias, TypedDict, TypeVar

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
    tol: float = 0.05,
    s_w: int = 3,
    r_w: int = 20,
    b_factor: float = 0.6,
    rng: RNGInput = None,
) -> dict[str, float]:
    """
    Summarize one or more statistics using the Bag of Little Bootstraps :cite:p:`blb`.
    """
    if stat != "mean":
        raise ValueError(f"unsupported statistic {stat}")

    mask = np.isfinite(xs)
    if ninf := int(np.sum(~mask)):
        warnings.warn(f"ignoring {ninf} nonfinite values", DataWarning, stacklevel=2)

    xs = xs[mask]
    est = np.average(xs).item()

    rng = random_generator(rng)
    bootstrapper = _BLBootstrapper(np.average, ci_width, tol, s_w, r_w, b_factor, rng)

    boot_df = bootstrapper.summarize(xs)

    return {"value": est} | boot_df.agg("mean").to_dict()


class _BootResult(TypedDict):
    mean: float
    ci_min: float
    ci_max: float
    count: int


class _BLBootstrapper:
    """
    Implementation of BLB computation.
    """

    statistic: WeightedStatistic
    ci_width: float

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

    def summarize(self, xs: NDArray[F]):
        results = []
        means = StatAccum(self.tolerance, self.s_window)

        count = 0
        for ss in self.blb_subsets(xs):
            count += 1
            res = self.measure_subset(xs, ss)
            results.append(res)
            means.record(res["mean"])
            if means.converged():
                break

        return pd.DataFrame.from_records(results)

    def blb_subsets(self, xs: NDArray[F]):
        b = int(len(xs) ** self.b_factor)

        while True:
            yield self.rng.choice(len(xs), b, replace=False)

    def measure_subset(self, xs: NDArray[F], ss: NDArray[np.int64]) -> _BootResult:
        b = len(ss)
        n = len(xs)
        xss = xs[ss]

        acc = StatAccum(self.tolerance, self.r_window)

        count = 0
        for weights in self.miniboot_weights(n, b):
            count += 1
            stat = self.statistic(xss, weights=weights)
            acc.record(stat)
            if acc.converged():
                break

        [lo, hi] = np.quantile(acc.values, [0.025, 0.975])
        return {"mean": np.mean(acc.values).item(), "ci_min": lo, "ci_max": hi, "count": count}

    def miniboot_weights(self, n: int, b: int):
        flat = np.full(b, 1.0 / b)

        while True:
            yield self.rng.multinomial(n, flat)


class StatAccum:
    INIT_SIZE: ClassVar[int] = 100
    ABS_TOL: ClassVar[float] = 1.0e-12

    tolerance: float
    window: int

    _len: int = 0
    _values: NDArray[np.float64]
    _cum_means: NDArray[np.float64]

    def __init__(self, tol: float, w: int):
        self.tolerance = tol
        self.window = w

        self._values = np.zeros(self.INIT_SIZE)
        self._cum_means = np.zeros(self.INIT_SIZE)

    @property
    def values(self) -> NDArray[np.float64]:
        return self._values[: self._len]

    def record(self, x: float | np.floating[Any]) -> None:
        "Record a new value in the accumulator."
        self._expand_if_needed()
        i = self._len
        self._len += 1

        # record and update the cumulative mean
        self._values[i] = x
        self._cum_means[i] = np.mean(self.values)

    def mean(self) -> float | None:
        "Get the mean of the accumulated values."
        if self._len > 0:
            return self._cum_means[self._len - 1]
        else:
            return None

    def converged(self) -> bool:
        """
        Check for convergence.
        """
        if self._len < self.window:
            return False

        i_cur = self._len - 1
        i_start = self._len - self.window
        current = self._cum_means[i_cur]

        # lower-bound tolerance for very small values
        atol = max(current * self.tolerance, self.ABS_TOL)

        window = self._cum_means[i_start : self._len]
        gaps = np.abs(window - current)
        return np.all(gaps <= atol).item()

    def _expand_if_needed(self):
        cap = len(self._values)
        if cap == self._len:
            self._values.resize(cap * 2)
            self._cum_means.resize(cap * 2)
