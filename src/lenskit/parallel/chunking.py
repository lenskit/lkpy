# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Support for computing chunks for parallel computation.
"""

# some defaults for chunking behavior
import math
from typing import NamedTuple

MIN_CHUNKABLE = 100
TGT_CHUNK_LIMIT = 1000
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 4000


class WorkChunks(NamedTuple):
    """
    The chunking configuration for parallel work.
    """

    total: int
    chunk_size: int
    chunk_count: int

    @classmethod
    def create(cls, njobs):
        """
        Compute the chunk size for parallel model training.
        """
        if njobs < MIN_CHUNKABLE:
            csize = njobs
            count = 1
        else:
            csize = max(njobs // TGT_CHUNK_LIMIT, MIN_CHUNK_SIZE)
            if csize > MAX_CHUNK_SIZE:
                csize = MAX_CHUNK_SIZE

            count = int(math.ceil(njobs / csize))

        return cls(njobs, csize, count)
