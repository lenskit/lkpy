# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass
from threading import Condition, Lock, Thread
from typing import Any, ClassVar, Deque, Literal, Protocol, TypeAlias, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from lenskit.diagnostics import DataWarning
from lenskit.logging import Tracer, get_logger, get_tracer, trace
from lenskit.random import RNGInput, random_generator

F = TypeVar("F", bound=np.floating, covariant=True)

SummaryStat: TypeAlias = Literal["mean"]

_log = get_logger(__name__)

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
        "rep_mean": result.rep_mean,
        "rep_var": result.rep_var,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
    }

    return result


@dataclass
class _BootResult:
    rep_mean: float
    rep_var: float
    ci_lower: float
    ci_upper: float
    samples: pd.DataFrame


class _BLBootstrapper:
    """
    Implementation of BLB computation.
    """

    _tracer: Tracer
    statistic: WeightedStatistic
    ci_width: float
    _ci_qmin: float
    _ci_qmax: float

    tolerance: float
    s_window: int
    r_window: int
    b_factor: float
    rng: np.random.Generator
    _rep_generator: ReplicateGenerator

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
        self._tracer = get_tracer(_log, stat=stat.__name__)  # type: ignore

    def run_bootstraps(self, xs: NDArray[F]) -> _BootResult:
        n = len(xs)
        b = int(n**self.b_factor)

        self._tracer.add_bindings(n=n, b=b)
        _log.debug("starting bootstrap", stat=self.statistic.__name__, n=len(xs))  # type: ignore
        ss_frames = {}

        means = StatAccum(np.mean)
        vars = StatAccum(np.mean)
        lbs = StatAccum(np.mean)
        ubs = StatAccum(np.mean)

        self._rep_generator = ReplicateGenerator(n, b, self.rng)
        self._tracer.trace("let's go!")

        with self._rep_generator:
            for i, ss in enumerate(self.blb_subsets(n, b)):
                self._tracer.add_bindings(subset=i)
                self._tracer.trace("starting subset")
                res = self.measure_subset(xs, ss)
                ss_frames[i] = res.samples
                means.record(res.rep_mean)
                vars.record(res.rep_var)
                lbs.record(res.ci_lower)
                ubs.record(res.ci_upper)
                if self._check_convergence(
                    means, vars, lbs, ubs, tol=self.tolerance, w=self.s_window
                ):
                    break

        return _BootResult(
            means.statistic,
            vars.statistic,
            lbs.statistic,
            ubs.statistic,
            pd.concat(ss_frames, names=["subset"]),
        )

    def blb_subsets(self, n: int, b: int):
        while True:
            yield self.rng.choice(n, b, replace=False)

    def measure_subset(self, xs: NDArray[F], ss: NDArray[np.int64]) -> _BootResult:
        b = len(ss)
        n = len(xs)
        xss = xs[ss]

        means = StatAccum(np.mean)
        vars = StatAccum(np.var)
        lbs = StatAccum(lambda a: np.quantile(a, self._ci_qmin))
        ubs = StatAccum(lambda a: np.quantile(a, self._ci_qmax))

        loop = self._rep_generator.subsets()
        for i, weights in enumerate(loop):
            self._tracer.add_bindings(rep=i)
            self._tracer.trace("starting replicate")
            assert weights.shape == (b,)
            assert np.sum(weights) == n
            stat = self.statistic(xss, weights=weights)
            means.record(stat)
            vars.record(stat)
            lbs.record(stat)
            ubs.record(stat)

            if self._check_convergence(means, vars, lbs, ubs, tol=self.tolerance, w=self.r_window):
                loop.close()

        df = pd.DataFrame({"statistic": means.values})
        df.index.name = "iter"
        self._tracer.remove_bindings("rep")
        return _BootResult(means.statistic, vars.statistic, lbs.statistic, ubs.statistic, df)

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


class ReplicateGenerator:
    """
    Generate the subset samples for a bootstrap in a background thread.
    """

    n: int
    b: int

    _rng: np.random.Generator
    _flat: NDArray[np.float64]
    _lock: Lock
    _notify: Condition
    _running: bool = True
    _queue: Deque
    _thread: Thread

    def __init__(self, n: int, b: int, rng: np.random.Generator):
        self.n = n
        self.b = b
        self._rng = rng.spawn(1)[0]
        self._queue = deque()
        self._flat = np.full(b, 1.0 / b)
        self._lock = Lock()
        self._notify = Condition(self._lock)

    def subsets(self) -> Generator[NDArray[np.int64], None, None]:
        while True:
            with self._notify:
                while self._thread.is_alive() and len(self._queue) == 0:
                    self._notify.wait()

                try:
                    val = self._queue.popleft()
                    self._notify.notify_all()
                except IndexError:
                    break  # things have shut down, loop is over
                except GeneratorExit:
                    break  # we've been asked to close

            yield val

    def _generate(self):
        with self._notify:
            while True:
                # check if we need to wake up
                while self._running and len(self._queue) >= 5:
                    trace(_log, "waiting for queue", len=len(self._queue))
                    self._notify.wait()

                # are we done?
                if not self._running:
                    break

                # generate a new value
                val = self._rng.multinomial(self.n, self._flat)
                self._queue.append(val)
                self._notify.notify_all()

    def __enter__(self):
        self._thread = Thread(target=self._generate)
        self._thread.start()
        return self

    def __exit__(self, *args: Any):
        with self._notify:
            self._running = False
            self._notify.notify_all()


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
