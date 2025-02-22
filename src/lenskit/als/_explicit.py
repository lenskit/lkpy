# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import torch
from typing_extensions import override

from lenskit.basic import BiasModel, Damping
from lenskit.data import Dataset, ItemList
from lenskit.logging.progress import item_progress_handle, pbh_update
from lenskit.math.solve import solve_cholesky
from lenskit.parallel.chunking import WorkChunks

from ._common import ALSBase, ALSConfig, TrainContext, TrainingData


class BiasedMFConfig(ALSConfig):
    damping: Damping = 5.0
    """
    Damping for the bias model.
    """


class BiasedMFScorer(ALSBase):
    """
    Biased matrix factorization trained with alternating least squares
    :cite:p:`zhouLargeScaleParallelCollaborative2008`.  This is a
    prediction-oriented algorithm suitable for explicit feedback data, using the
    alternating least squares approach to compute :math:`P` and :math:`Q` to
    minimize the regularized squared reconstruction error of the ratings matrix.

    See the base class :class:`ALSBase` for documentation on the estimated
    parameters you can extract from a trained model. See
    :class:`BiasedMFConfig` and :class:`ALSConfig` for the configuration
    options for this component.

    Stability:
        Caller
    """

    config: BiasedMFConfig
    bias_: BiasModel

    @override
    def prepare_data(self, data: Dataset):
        # transform ratings using offsets
        rmat = data.interaction_matrix(format="torch", layout="coo", field="rating")

        self.logger.info("normalizing ratings")
        self.bias_ = BiasModel.learn(data, damping=self.config.damping)
        rmat = self.bias_.transform_matrix(rmat)

        rmat = rmat.to_sparse_csr()
        assert not torch.any(torch.isnan(rmat.values()))
        return TrainingData.create(data.users, data.items, rmat)

    @override
    def initial_params(self, nrows: int, ncols: int, rng: np.random.Generator) -> torch.Tensor:
        mat = rng.standard_normal((nrows, ncols))
        mat /= np.linalg.norm(mat, axis=1).reshape((nrows, 1))
        mat = torch.from_numpy(mat)
        return mat

    @override
    def als_half_epoch(self, epoch: int, context: TrainContext):
        chunks = WorkChunks.create(context.nrows)
        with item_progress_handle(f"epoch {epoch} {context.label}s", total=context.nrows) as pbh:
            return _train_update_fanout(context, chunks, pbh)

    @override
    def new_user_embedding(
        self, user_num: int | None, items: ItemList
    ) -> tuple[torch.Tensor, float | None]:
        inums = items.numbers("torch", vocabulary=self.items_, missing="negative")
        mask = inums >= 0
        ratings = items.field("rating", "torch")
        assert ratings is not None

        biases, u_bias = self.bias_.compute_for_items(items, None, items)
        biases = torch.from_numpy(biases)
        ratings = ratings - biases

        ri_val = ratings[mask].to(torch.float64)

        u_feat = _train_bias_row_cholesky(
            inums[mask], ri_val, self.item_features_, self.config.user_reg
        )
        return u_feat, u_bias

    @override
    def finalize_scores(
        self, user_num: int | None, items: ItemList, user_bias: float | None
    ) -> ItemList:
        scores = items.scores()
        assert scores is not None

        if user_bias is None:
            if user_num is not None and self.bias_.user_biases is not None:
                user_bias = self.bias_.user_biases[user_num]
            else:
                user_bias = 0.0
        assert user_bias is not None

        biases = self.bias_.compute_for_items(items, bias=user_bias)
        scores = scores + biases

        return ItemList(items, scores=scores)


@torch.jit.script
def _train_solve_row(
    cols: torch.Tensor,
    vals: torch.Tensor,
    this: torch.Tensor,
    other: torch.Tensor,
    regI: torch.Tensor,
) -> torch.Tensor:
    (nui,) = cols.shape
    nf = other.shape[1]

    if nui == 0:
        return torch.zeros(nf)

    M = other[cols, :]
    MMT = M.T @ M
    # assert MMT.shape[0] == ctx.n_features
    # assert MMT.shape[1] == ctx.n_features
    A = MMT + regI * nui
    V = M.T @ vals
    # and solve
    return solve_cholesky(A, V)


@torch.jit.script
def _train_update_rows(ctx: TrainContext, start: int, end: int, pbh: str) -> torch.Tensor:
    result = ctx.left[start:end, :].clone()

    for i in range(start, end):
        row = ctx.matrix[i]
        (n,) = row.shape
        if n == 0:
            continue

        cols = row.indices()[0]
        vals = row.values().type(ctx.left.type())

        V = _train_solve_row(cols, vals, ctx.left, ctx.right, ctx.regI)
        result[i - start] = V
        pbh_update(pbh, 1)

    return result


@torch.jit.script
def _train_update_fanout(ctx: TrainContext, chunking: WorkChunks, pbh: str) -> float:
    if ctx.nrows <= 50:
        # at 50 rows, we run sequentially
        M = _train_update_rows(ctx, 0, ctx.nrows, pbh)
        sqerr = torch.norm(ctx.left - M)
        ctx.left[:, :] = M
        return sqerr.item()

    results: list[tuple[int, int, torch.jit.Future[torch.Tensor]]] = []
    for start in range(0, ctx.nrows, chunking.chunk_size):
        end = min(start + chunking.chunk_size, ctx.nrows)
        results.append((start, end, torch.jit.fork(_train_update_rows, ctx, start, end, pbh)))  # type: ignore

    sqerr = torch.tensor(0.0)
    for start, end, r in results:
        M = r.wait()
        diff = (ctx.left[start:end, :] - M).ravel()
        sqerr += torch.dot(diff, diff)
        ctx.left[start:end, :] = M

    return sqerr.sqrt().item()


def _train_bias_row_cholesky(
    items: torch.Tensor, ratings: torch.Tensor, other: torch.Tensor, reg: float
) -> torch.Tensor:
    """
    Args:
        items: the item IDs the user has rated
        ratings: the user's (normalized) ratings for those items
        other: the item-feature matrix
        reg: the regularization term
    Returns:
        the user-feature vector (equivalent to V in the current Cholesky code)
    """
    (nui,) = items.shape
    nf = other.shape[1]

    if nui == 0:
        return torch.zeros(nf)

    M = other[items, :]
    regI = torch.eye(nf, device=other.device) * reg
    MMT = M.T @ M
    A = MMT + regI * len(items)

    V = M.T @ ratings
    x = solve_cholesky(A, V)

    return x
