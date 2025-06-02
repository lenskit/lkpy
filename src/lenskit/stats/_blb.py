# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any, Literal, Protocol, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

from lenskit.diagnostics import DataWarning
from lenskit.random import Generator, RNGInput, random_generator

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
    s: int = 10,
    r: int = 100,
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
    n = len(xs)
    b = int(n**b_factor)

    rng = random_generator(rng)

    ss_summaries = {}
    for ss in _blb_subsets(xs, s, b, rng=rng):
        ss_sum = _miniboot_ss(xs, ss, np.average, r, rng=rng)
        _accum_summaries(ss_sum, dest=ss_summaries)

    return {"value": est} | {n: np.mean(xs) for n, xs in ss_summaries.items()}


def _blb_subsets(xs: NDArray[F], s: int, b: int, *, rng: Generator) -> Iterable[NDArray[np.int64]]:
    for i in range(s):
        yield rng.choice(len(xs), b, replace=False)


def _accum_summaries(values: dict[str, float], *, dest: dict[str, list[float]]):
    for name, value in values.items():
        vs = dest.setdefault(name, [])
        vs.append(value)


def _miniboot_ss(
    xs: NDArray[F], ss: NDArray[np.int64], stat: WeightedStatistic, r: int, *, rng: Generator
) -> dict[str, float]:
    b = len(ss)
    n = len(xs)
    xss = xs[ss]

    flat = np.full(b, 1.0 / b)
    vals = [_miniboot_sample_stat(n, xss, flat, stat, rng) for _j in range(r)]
    vals = np.array(vals)
    mean = np.mean(vals).item()
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {"mean": mean, "low": lo, "high": hi}


def _miniboot_sample_stat(
    n: int, xss: NDArray[F], flat: NDArray[np.float64], stat: WeightedStatistic, rng: Generator
) -> float:
    weights = rng.multinomial(n, flat)
    assert weights.shape == (len(flat),)
    assert np.sum(weights) == n
    return stat(xss, weights=weights).item()
