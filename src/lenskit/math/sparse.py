# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Sparse matrix utility functions.
"""

# pyright: basic
from __future__ import annotations

import logging
from typing import Literal

import scipy.sparse as sps
import torch

from lenskit.torch import safe_tensor

_log = logging.getLogger(__name__)


def torch_sparse_from_scipy(
    M: sps.coo_array, layout: Literal["csr", "coo", "csc"] = "coo"
) -> torch.Tensor:
    """
    Convert a SciPy :class:`sps.coo_array` into a torch sparse tensor.

    Stability:
        Internal
    """
    ris = safe_tensor(M.row)
    cis = safe_tensor(M.col)
    vs = safe_tensor(M.data)
    indices = torch.stack([ris, cis])
    assert indices.shape == (2, M.nnz)
    T = torch.sparse_coo_tensor(indices, vs, size=M.shape)
    assert T.shape == M.shape

    match layout:
        case "csr":
            return T.to_sparse_csr()
        case "csc":
            return T.to_sparse_csc()
        case "coo":
            return T.coalesce()
        case _:
            raise ValueError(f"invalid layout {layout}")
