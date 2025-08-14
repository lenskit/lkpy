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
import scipy.stats
from numpy.typing import NDArray

from lenskit.diagnostics import DataWarning
from lenskit.logging import Tracer, get_logger, get_tracer, trace
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
    _rep_generator: ReplicateGenerator

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

        self._tracer.trace("estimating acceleration term")
        accel = _bca_accel_term(xs, self.config.statistic)

        self._rep_generator = ReplicateGenerator(n, b, self.rng)
        self._tracer.trace("let's go!")

        with self._rep_generator:
            for i, ss in enumerate(self.blb_subsets(n, b)):
                self._tracer.add_bindings(subset=i)
                self._tracer.trace("starting subset")
                res = self.measure_subset(xs, ss, estimate, accel)
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

    def measure_subset(
        self, xs: NDArray[F], ss: NDArray[np.int64], estimate: float, accel: float
    ) -> _BootResult:
        b = len(ss)
        n = len(xs)
        xss = xs[ss]

        means = StatAccum(np.mean)
        vars = StatAccum(np.var)
        lbs = StatAccum(None)
        ubs = StatAccum(None)

        loop = self.miniboot_weights(n, b)
        for i, weights in enumerate(loop):
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
                loop.close()

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


def _bca_range(
    estimate: float, replicates: NDArray[np.floating[Any]], margin: float, accel: float
) -> tuple[float, float]:
    """
    Estimate the BCa quantiles for a bootstrap.

    This follows Slide 34 of `http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf`_.
    """
    bias = _bca_bias_corrector(estimate, replicates)
    trace(_log, "B=%d, estimate=%f, bias=%f", len(replicates), estimate, bias)

    z1 = bias + STD_NORM.ppf(margin)
    icd1 = z1 / (1 - accel * z1)

    z2 = bias + STD_NORM.ppf(1 - margin)
    icd2 = z2 / (1 - accel * z2)

    return STD_NORM.cdf(icd1), STD_NORM.cdf(icd2)


def _bca_bias_corrector(statistic: float, replicates: NDArray[np.floating[Any]]) -> float:
    B = len(replicates)
    nlow = np.sum(replicates < statistic)
    if nlow == 0 or nlow == B:
        # extremely biased, but goes OOB. Should only happen early in the bootstrap.
        return 0
    else:
        return STD_NORM.ppf(nlow / B)


def _bca_accel_term(xs: NDArray[np.floating[Any]], statistic: WeightedStatistic) -> float:
    """
    Compute the BCa acceleration term.

    Follows slide 36 of
    `http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf`_, referring also
    to the SciPy `scipy/stats/_resampling.py` for implementation ideas.
    """
    N = len(xs)
    BSIZE = 5000
    jk_vals = np.empty(N)
    # batch the jackknife, because our data might be huge
    # TODO: can we sample the jackknife?
    for start in range(0, N, BSIZE):
        end = min(start + BSIZE, N)
        B = end - start
        # this trick is from scipy — set up a mask
        mask = np.ones((B, N), dtype=np.bool_)
        np.fill_diagonal(mask[:, start:end], False)
        # and reshape — again, borrwed from scipy
        i = np.broadcast_to(np.arange(N), (B, N))
        i = i[mask].reshape((B, N - 1))

        # prepare B x N batched sample and compute statistics
        sample = xs[i]
        stats = statistic(sample, axis=-1)
        assert stats.shape == (B,)
        jk_vals[start:end] = stats

    jk_est = np.mean(jk_vals)
    jk_dev = jk_est - jk_vals

    # sum of cubes
    accel_num = np.sum(np.power(jk_dev, 3))
    # weird term
    accel_denom = 6 * np.power(np.sum(np.square(jk_dev)), 1.5)
    return accel_num / accel_denom
