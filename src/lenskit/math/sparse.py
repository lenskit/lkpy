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
from typing import Literal, overload

import scipy.sparse as sps
import torch

_log = logging.getLogger(__name__)


@overload
def normalize_sparse_rows(
    matrix: torch.Tensor, method: Literal["center"], inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]: ...
@overload
def normalize_sparse_rows(
    matrix: torch.Tensor, method: Literal["unit"], inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]: ...
def normalize_sparse_rows(
    matrix: torch.Tensor, method: str, inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize the rows of a sparse matrix.
    """
    match method:
        case "unit":
            return _nsr_unit(matrix)
        case "center":
            return _nsr_mean_center(matrix)
        case _:
            raise ValueError(f"unsupported normalization method {method}")


def _nsr_mean_center(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    nr, _nc = matrix.shape
    sums = matrix.sum(dim=1, keepdim=True).to_dense().reshape(nr)
    counts = torch.diff(matrix.crow_indices())
    assert sums.shape == counts.shape
    means = torch.nan_to_num(sums / counts, 0)
    return torch.sparse_csr_tensor(
        crow_indices=matrix.crow_indices(),
        col_indices=matrix.col_indices(),
        values=matrix.values() - torch.repeat_interleave(means, counts),
        size=matrix.shape,
    ), means


def _nsr_unit(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sqmat = torch.sparse_csr_tensor(
        crow_indices=matrix.crow_indices(),
        col_indices=matrix.col_indices(),
        values=matrix.values().square(),
    )
    norms = sqmat.sum(dim=1, keepdim=True).to_dense().reshape(matrix.shape[0])
    norms.sqrt_()
    recip_norms = torch.where(norms > 0, torch.reciprocal(norms), 0.0)
    return torch.sparse_csr_tensor(
        crow_indices=matrix.crow_indices(),
        col_indices=matrix.col_indices(),
        values=matrix.values() * torch.repeat_interleave(recip_norms, matrix.crow_indices().diff()),
        size=matrix.shape,
    ), norms


def torch_sparse_from_scipy(
    M: sps.coo_array, layout: Literal["csr", "coo", "csc"] = "coo"
) -> torch.Tensor:
    """
    Convert a SciPy :class:`sps.coo_array` into a torch sparse tensor.
    """
    ris = torch.from_numpy(M.row)
    cis = torch.from_numpy(M.col)
    vs = torch.from_numpy(M.data)
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


def torch_sparse_to_scipy(M: torch.Tensor) -> sps.csr_array | sps.coo_array:
    if M.is_sparse:
        return sps.coo_array((M.values().numpy(), M.indices().numpy()), shape=M.shape)
    elif M.is_sparse_csr:
        return sps.csr_array(
            (M.values().numpy(), M.col_indices().numpy(), M.crow_indices().numpy()), shape=M.shape
        )
    else:
        raise TypeError("unsupported tensor type")
