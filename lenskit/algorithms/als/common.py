from __future__ import annotations

from typing import NamedTuple

import pandas as pd
import torch


class TrainContext(NamedTuple):
    """
    Context objedct for one half of an ALS training operation.
    """

    matrix: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    reg: float
    nrows: int
    ncols: int
    embed_size: int
    regI: torch.Tensor

    @classmethod
    def create(
        cls, matrix: torch.Tensor, left: torch.Tensor, right: torch.Tensor, reg: float
    ) -> TrainContext:
        nrows, ncols = matrix.shape
        lnr, embed_size = left.shape
        assert lnr == nrows
        assert right.shape == (ncols, embed_size)
        regI = torch.eye(embed_size, dtype=left.dtype, device=left.device)
        return TrainContext(matrix, left, right, reg, nrows, ncols, embed_size, regI)


class PartialModel(NamedTuple):
    """
    Partially-trained matrix factorization model.
    """

    users: pd.Index
    items: pd.Index
    user_matrix: torch.Tensor
    item_matrix: torch.Tensor
