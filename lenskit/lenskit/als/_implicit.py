# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import numpy as np
import torch
from typing_extensions import override

from lenskit.data import Dataset, ItemList
from lenskit.logging import get_logger
from lenskit.logging.progress import item_progress_handle, pbh_update
from lenskit.math.solve import solve_cholesky
from lenskit.parallel.chunking import WorkChunks

from ._common import ALSBase, ALSConfig, TrainContext, TrainingData


class ImplicitMFConfig(ALSConfig):
    weight: float = 40
    """
    The confidence weight for positive examples.
    """
    use_ratings: bool = False
    """
    If ``True``, use rating values instead of just presence or absence.
    """


class ImplicitMFScorer(ALSBase):
    """
    Implicit matrix factorization trained with alternating least squares
    :cite:p:`hu:implicit-mf`.  This algorithm outputs 'predictions', but they
    are not on a meaningful scale.  If its input data contains ``rating``
    values, these will be used as the 'confidence' values; otherwise, confidence
    will be 1 for every rated item.

    See the base class :class:`.MFPredictor` for documentation on the estimated
    parameters you can extract from a trained model.

    With weight :math:`w`, this function decomposes the matrix
    :math:`\\mathbb{1}^* + Rw`, where :math:`\\mathbb{1}^*` is an :math:`m
    \\times n` matrix of all 1s.

    .. versionchanged:: 2025.1
        ``ImplicitMFScorer`` no longer supports multiple training methods. It always uses
        Cholesky decomposition now.

    .. versionchanged:: 0.14
        By default, ``ImplicitMF`` ignores a ``rating`` column if one is present in the training
        data.  This can be changed through the ``use_ratings`` option.

    .. versionchanged:: 0.13
        In versions prior to 0.13, ``ImplicitMF`` used the rating column if it was present.
        In 0.13, we added an option to control whether or not the rating column is used; it
        initially defaulted to ``True``, but with a warning.  In 0.14 it defaults to ``False``.

    Stability:
        Caller
    """

    logger = get_logger(__name__, variant="implicit")

    config: ImplicitMFConfig

    OtOr_: torch.Tensor

    @override
    def train(self, data: Dataset):
        super().train(data)

        # compute OtOr and save it on the model
        reg = self.config.user_reg
        self.OtOr_ = _implicit_otor(self.item_features_, reg)

    @override
    def prepare_data(self, data: Dataset) -> TrainingData:
        if self.config.use_ratings:
            rmat = data.interaction_matrix("torch", field="rating")
        else:
            rmat = data.interaction_matrix("torch")

        rmat = torch.sparse_csr_tensor(
            crow_indices=rmat.crow_indices(),
            col_indices=rmat.col_indices(),
            values=rmat.values() * self.config.weight,
            size=rmat.shape,
        )
        return TrainingData.create(data.users, data.items, rmat)

    @override
    def initial_params(self, nrows: int, ncols: int, rng: np.random.Generator) -> torch.Tensor:
        mat = rng.standard_normal((nrows, ncols)) * 0.01
        mat = torch.from_numpy(mat)
        mat.square_()
        return mat

    @override
    def als_half_epoch(self, epoch: int, context: TrainContext) -> float:
        chunks = WorkChunks.create(context.nrows)

        OtOr = _implicit_otor(context.right, context.reg)
        with item_progress_handle(f"epoch {epoch} {context.label}s", total=context.nrows) as pbh:
            return _train_implicit_cholesky_fanout(context, OtOr, chunks, pbh)

    @override
    def new_user_embedding(
        self, user_num: int | None, user_items: ItemList
    ) -> tuple[torch.Tensor, None]:
        ri_idxes = user_items.numbers("torch", vocabulary=self.items_)

        ri_good = ri_idxes >= 0
        ri_it = ri_idxes[ri_good]
        if self.config.use_ratings:
            ratings = user_items.field("rating", "torch")
            if ratings is None:
                raise ValueError("no ratings in user items")
            ri_val = ratings[ri_good] * self.config.weight
        else:
            ri_val = torch.full((len(ri_good),), self.config.weight)

        ri_val = ri_val.to(self.item_features_.dtype)

        u_feat = self._train_new_row(ri_it, ri_val, self.item_features_, self.OtOr_)
        return u_feat, None

    def _train_new_row(
        self, items: torch.Tensor, ratings: torch.Tensor, i_embeds: torch.Tensor, OtOr: torch.Tensor
    ) -> torch.Tensor:
        """
        Train a single user row with new rating data.

        Args:
            items: the item IDs the user has rated
            ratings: the user's ratings for those items (when rating values are used)
            other: the item-feature matrix
            OtOr: the pre-computed regularization and background matrix.

        Returns:
            The user-feature vector.
        """
        self.logger.debug("learning new user row", n_items=len(items))

        # we can optimize by only considering the nonzero entries of Cu-I
        # this means we only need the corresponding matrix columns
        M = i_embeds[items, :]
        # Compute M^T (C_u-I) M, restricted to these nonzero entries
        MMT = (M.T * ratings) @ M
        # Build the matrix for solving
        A = OtOr + MMT
        # Compute RHS - only used columns (p_ui != 0) values needed
        y = i_embeds.T[:, items] @ (ratings + 1.0)
        # and solve
        x = solve_cholesky(A, y)

        return x


@torch.jit.script
def _implicit_otor(other: torch.Tensor, reg: float) -> torch.Tensor:
    nf = other.shape[1]
    regmat = torch.eye(nf)
    regmat *= reg
    Ot = other.T
    OtO = Ot @ other
    OtO += regmat
    return OtO


def _train_implicit_cholesky_rows(
    ctx: TrainContext, OtOr: torch.Tensor, start: int, end: int, pbh: str
) -> torch.Tensor:
    result = ctx.left[start:end, :].clone()

    for i in range(start, end):
        row = ctx.matrix[i]
        (n,) = row.shape
        if n == 0:
            continue

        cols = row.indices()[0]
        vals = row.values().type(ctx.left.type())

        # we can optimize by only considering the nonzero entries of Cu-I
        # this means we only need the corresponding matrix columns
        M = ctx.right[cols, :]
        # Compute M^T (C_u-I) M, restricted to these nonzero entries
        MMT = (M.T * vals) @ M
        # assert MMT.shape[0] == ctx.n_features
        # assert MMT.shape[1] == ctx.n_features
        # Build the matrix for solving
        A = OtOr + MMT
        # Compute RHS - only used columns (p_ui != 0) values needed
        # Cu is rates + 1 for the cols, so just trim Ot
        y = ctx.right.T[:, cols] @ (vals + 1.0)
        # and solve
        x = solve_cholesky(A, y)
        # assert len(uv) == ctx.n_features
        result[i - start, :] = x

        pbh_update(pbh, 1)

    return result


@torch.jit.script
def _train_implicit_cholesky_fanout(
    ctx: TrainContext, OtOr: torch.Tensor, chunks: WorkChunks, pbh: str
) -> float:
    results: list[tuple[int, int, torch.jit.Future[torch.Tensor]]] = []
    for start in range(0, ctx.nrows, chunks.chunk_size):
        end = min(start + chunks.chunk_size, ctx.nrows)
        results.append(
            (start, end, torch.jit.fork(_train_implicit_cholesky_rows, ctx, OtOr, start, end, pbh))  # type: ignore
        )

    sqerr = torch.tensor(0.0)
    for start, end, r in results:
        M = r.wait()
        diff = (ctx.left[start:end, :] - M).ravel()
        sqerr += torch.dot(diff, diff)
        ctx.left[start:end, :] = M

    return math.sqrt(sqerr.item())
