# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Data manipulation routines.
"""

# pyright: basic
from __future__ import annotations

from typing import NamedTuple, TypeVar

import numpy as np
import scipy.sparse as sps
import torch

t = torch
M = TypeVar("M", "CSRStructure", sps.csr_array, sps.coo_array, sps.spmatrix, t.Tensor)


class CSRStructure(NamedTuple):
    """
    Representation of the compressed sparse row structure of a sparse matrix,
    without any data values.

    Stability:
        Caller
    """

    rowptrs: np.ndarray
    colinds: np.ndarray
    shape: tuple[int, int]

    @property
    def nrows(self):
        return self.shape[0]

    @property
    def ncols(self):
        return self.shape[1]

    @property
    def nnz(self):
        return self.rowptrs[self.nrows]

    def extent(self, row: int) -> tuple[int, int]:
        return self.rowptrs[row], self.rowptrs[row + 1]

    def row_cs(self, row: int) -> np.ndarray:
        sp, ep = self.extent(row)
        return self.colinds[sp:ep]
