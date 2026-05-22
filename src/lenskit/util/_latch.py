# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit._accel import AtomicInt


class Latch:
    """
    A thread-safe “latch” that can be tripped and reset.
    """

    _counter: AtomicInt

    def __init__(self):
        self._counter = AtomicInt()

    def is_latched(self):
        return self._counter.load() > 0

    def latch(self) -> bool:
        """
        Latch the latch, returning ``True`` if this call took the latch, and
        ``False`` if it was already latched.
        """
        x = self._counter.fetch_add()
        return x == 0

    def reset(self):
        """
        Reset the latch.
        """
        self._counter.store(0)
