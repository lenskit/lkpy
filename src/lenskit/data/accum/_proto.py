# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AccumulatorFactory[X, R](Protocol):
    def create_accumulator(self) -> Accumulator[X, R]:
        """
        Create an accumulator for the results of this object.

        Return:
            An accumulator.
        """
        ...


@runtime_checkable
class Accumulator[X, R](Protocol):
    """
    Protocol implemented by data accumulators.
    """

    def add(self, value: X) -> None:
        """
        Add a single value to this accumulator.
        """
        ...

    def accumulate(self) -> R:
        """
        Compute the accumulated value from this accumulator.
        """
        ...
