# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from ._proto import Accumulator


class ObjectListAccumulator[T](Accumulator[T, list[T]]):
    """
    An accumulator lists of objects.
    """

    values: list[T]

    def __init__(self):
        self.values = []

    def __len__(self) -> int:
        return len(self.values)

    def add(self, value: T):
        self.values.append(value)

    def accumulate(self) -> list[T]:
        return self.values
