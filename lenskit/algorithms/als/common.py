from __future__ import annotations

import math
from collections import namedtuple
from typing import Literal, NamedTuple, TypeAlias

import torch

TrainHalf: TypeAlias = Literal["left", "right"]
PartialModel = namedtuple("PartialModel", ["users", "items", "user_matrix", "item_matrix"])


def train_chunking(nrows: int, max_chunks: int = 1024, min_size: int = 20):
    if nrows < 50:
        csize = nrows
        count = 1
    else:
        csize = max(nrows // max_chunks, min_size)
        count = int(math.ceil(nrows / csize))

    return TrainChunking(csize, count)


class TrainChunking(NamedTuple):
    chunk_size: int
    chunk_count: int


class TrainContext(NamedTuple):
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
