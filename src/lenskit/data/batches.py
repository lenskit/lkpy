# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Basic batching support.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

from lenskit.logging import get_logger

from .types import Extent

_log = get_logger(__name__)


@dataclass
class BatchIter:
    """
    Iterator over batches of a collection size.

    The length of this collection is the number of batches,
    and iterating over it yields batch extents.
    """

    n: int
    "The total number of items."
    size: int
    "The batch size."

    @property
    def batch_count(self) -> int:
        return math.ceil(self.n / self.size)

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self) -> Iterator[Extent]:
        _log.debug("iterating batches", N=self.n, count=self.batch_count, batch_size=self.size)
        for start in range(0, self.n, self.size):
            end = min(start + self.size, self.n)
            yield Extent(start, end)
