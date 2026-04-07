# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import TypedDict

import numpy as np

from ._proto import Accumulator

INITIAL_SIZE = 1024


class ValueStatistics(TypedDict):
    """
    Collected statitsics from :class:`ValueAccumulator`.
    """

    n: int
    mean: float
    median: float
    std: float


class ValueStatAccumulator(Accumulator[float | None, ValueStatistics]):
    """
    An accumulator for single real values, computing basic statistics.
    """

    _values: np.ndarray[tuple[int], np.dtype[np.float64]]
    _n: int = 0

    def __init__(self):
        self._values = np.empty(INITIAL_SIZE)
        self._n = 0

    @property
    def values(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._values[: self._n]

    def __len__(self) -> int:
        return self._n

    def add(self, value: float | None):
        if value is None or np.isnan(value):
            return

        if self._n == len(self._values):
            self._values.resize(self._n * 2)
        self._values[self._n] = value
        self._n += 1

    def accumulate(self) -> ValueStatistics:
        n = self._n
        return {
            "n": n,
            "mean": np.mean(self._values[:n]).item(),
            "median": np.median(self._values[:n]).item(),
            "std": np.std(self._values[:n]).item(),
        }
