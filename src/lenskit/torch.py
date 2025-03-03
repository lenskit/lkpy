# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
PyTorch utility functions.
"""

import functools

import torch


def inference_mode(func):
    """
    Function decorator that puts PyTorch in inference mode.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.inference_mode():
            return func(*args, **kwargs)

    return wrapper


def sparse_row(mat: torch.Tensor, row: int) -> torch.Tensor:
    """
    Get a row of a sparse (CSR) tensor.  This is needed because indexing a
    tensor does not work in inference mode.
    """

    assert mat.is_sparse_csr

    cri = mat.crow_indices()
    sp = cri[row]
    ep = cri[row + 1]

    cs = mat.col_indices()
    vs = mat.values()
    return torch.sparse_coo_tensor(
        indices=cs[sp:ep].reshape(1, -1), values=vs[sp:ep], size=mat.shape[1:]
    )
