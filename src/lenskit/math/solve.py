# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Efficient solver routines.
"""

import torch

from lenskit.data.types import NPMatrix, NPVector


def solve_cholesky(A: NPMatrix, y: NPVector) -> NPVector:
    """
    Solve the system :math:`A\\mathbf{x}=\\mathbf{y}` for :math:`\\mathbf{x}`
    with Cholesky decomposition.

    This wraps :func:`torch.linalg.cholesky_ex` and :func:`torch.cholesky_solve`
    in an easier-to-use interface with error checking.

    Args:
        A:
            the left-hand matrix :math:`A`
        y:
            the right-hand vector :math:`\\mathbf{y}`

    Returns:
        the solution :math:`\\mathbf{x}`
    """
    if len(y.shape) > 1:  # pragma: no cover
        raise TypeError(f"y must be 1D (found shape {y.shape})")
    (n,) = y.shape
    if A.shape != (n, n):  # pragma: no cover
        raise TypeError("A must be n⨉n")

    At = torch.from_numpy(A)
    yt = torch.from_numpy(y)

    L, info = torch.linalg.cholesky_ex(At)
    if info:
        raise RuntimeError("cholesky solve failed")
    yt = yt.reshape(1, n, 1)
    return torch.cholesky_solve(yt, L).reshape(n).numpy()
