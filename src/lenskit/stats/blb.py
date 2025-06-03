# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Implementation of the Bag of Little Bootstraps (BLB) method.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np
import pandas as pd

from lenskit.diagnostics import DataWarning
from lenskit.random import RNGInput, random_generator

T = TypeVar("T")
S = TypeVar("S")


@dataclass
class BLBResult:
    """
    Results from a BLB analysis.
    """

    mean: float
    "The mean of the statistic across all bootstrap replicates."
    std: float
    "The standard deviation of the statistic across all bootstrap replicates."
    ci_lower: float
    "The lower bound of the confidence interval."
    ci_upper: float
    "The upper bound of the confidence interval."
    replicates: np.ndarray
    "The individual replicate values."

    def __str__(self):
        return f"mean={self.mean:.4f} (±{self.std:.4f}, {self.ci_lower:.4f}—{self.ci_upper:.4f})"


def _process_subsample(
    data: np.ndarray | pd.DataFrame,
    statistic: Callable[[np.ndarray | pd.DataFrame], float],
    subsample_idx: np.ndarray,
    n: int,
    subsample_n: int,
    n_bootstrap: int,
    seed: int | None,
) -> np.ndarray:
    """
    Process a single subsample for BLB analysis.
    This is separated out to facilitate parallel processing.
    """
    rng = random_generator(seed)
    values = np.zeros(n_bootstrap)

    subsample = data[subsample_idx] if isinstance(data, np.ndarray) else data.iloc[subsample_idx]

    for j in range(n_bootstrap):
        boot_idx = rng.choice(subsample_n, size=n, replace=True)
        bootstrap = (
            subsample[boot_idx] if isinstance(data, np.ndarray) else subsample.iloc[boot_idx]
        )
        values[j] = statistic(bootstrap)

    return values


class BagOfLittleBootstraps:
    """
    Implementation of the Bag of Little Bootstraps method for statistical analysis.

    The BLB method works by:
    1. Taking multiple subsamples of the data
    2. For each subsample, computing bootstrap replicates
    3. Computing the desired statistic on each replicate
    4. Aggregating the results across all subsamples and replicates

    Args:
        n_subsamples:
            The number of subsamples to use (default: 10)
        subsample_size:
            The size of each subsample as a fraction of the data (default: 0.5)
        n_bootstrap:
            The number of bootstrap replicates per subsample (default: 100)
        confidence:
            The confidence level for intervals (default: 0.95)
        rng:
            Random number generator or seed
    """

    def __init__(
        self,
        n_subsamples: int = 10,
        subsample_size: float = 0.5,
        n_bootstrap: int = 100,
        confidence: float = 0.95,
        rng: RNGInput = None,
    ):
        self.n_subsamples = n_subsamples
        self.subsample_size = subsample_size
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self._rng = random_generator(rng)

    def analyze(
        self,
        data: np.ndarray | pd.DataFrame,
        statistic: Callable[[np.ndarray | pd.DataFrame], float],
    ) -> BLBResult:
        """
        Analyze a dataset using the BLB method (sequential implementation).

        Args:
            data:
                The input data (numpy array or pandas DataFrame)
            statistic:
                A function that computes the desired statistic on the data

        Returns:
            The BLB analysis results
        """
        n = len(data)
        subsample_n = int(n * self.subsample_size)
        alpha = 1 - self.confidence
        all_values = np.zeros(self.n_subsamples * self.n_bootstrap)

        for i in range(self.n_subsamples):
            subsample_idx = self._rng.choice(n, size=subsample_n, replace=False)
            values = _process_subsample(
                data,
                statistic,
                subsample_idx,
                n,
                subsample_n,
                self.n_bootstrap,
                self._rng.integers(0, 2**32),
            )
            all_values[i * self.n_bootstrap : (i + 1) * self.n_bootstrap] = values

        return self._compute_results(all_values)

    def analyze_parallel(
        self,
        data: np.ndarray | pd.DataFrame,
        statistic: Callable[[np.ndarray | pd.DataFrame], float],
    ) -> BLBResult:
        """
        Analyze a dataset using the BLB method with Ray-based parallelization.

        This method requires Ray to be installed and initialized. If Ray is not
        available, it falls back to the sequential implementation with a warning.

        Args:
            data:
                The input data (numpy array or pandas DataFrame)
            statistic:
                A function that computes the desired statistic on the data

        Returns:
            The BLB analysis results
        """
        try:
            import ray
        except ImportError:
            warnings.warn(
                "Ray is not available - falling back to sequential implementation",
                DataWarning,
                stacklevel=2,
            )
            return self.analyze(data, statistic)

        if not ray.is_initialized():
            warnings.warn(
                "Ray is not initialized - falling back to sequential implementation",
                DataWarning,
                stacklevel=2,
            )
            return self.analyze(data, statistic)

        n = len(data)
        subsample_n = int(n * self.subsample_size)

        @ray.remote
        def remote_process_subsample(
            data, statistic, subsample_idx, n, subsample_n, n_bootstrap, seed
        ):
            return _process_subsample(
                data, statistic, subsample_idx, n, subsample_n, n_bootstrap, seed
            )

        subsample_indices = [
            self._rng.choice(n, size=subsample_n, replace=False) for _ in range(self.n_subsamples)
        ]
        seeds = self._rng.integers(0, 2**32, size=self.n_subsamples)

        futures = [
            remote_process_subsample.remote(
                data, statistic, idx, n, subsample_n, self.n_bootstrap, seed
            )
            for idx, seed in zip(subsample_indices, seeds)
        ]

        results = ray.get(futures)
        all_values = np.concatenate(results)

        return self._compute_results(all_values)

    def _compute_results(self, all_values: np.ndarray) -> BLBResult:
        """
        Compute final BLB results from replicate values.
        """
        alpha = 1 - self.confidence
        mean = np.mean(all_values)
        std = np.std(all_values, ddof=1)
        ci_lower = np.percentile(all_values, alpha * 100 / 2)
        ci_upper = np.percentile(all_values, 100 - alpha * 100 / 2)

        return BLBResult(mean, std, ci_lower, ci_upper, all_values)
