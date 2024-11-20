# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging

import numpy as np
import torch
from seedbank import SeedLike
from typing_extensions import override

from lenskit.basic import BiasModel
from lenskit.data import Dataset, ItemList
from lenskit.data.types import UITuple
from lenskit.math.solve import solve_cholesky
from lenskit.parallel.chunking import WorkChunks
from lenskit.util.logging import pbh_update, progress_handle

from ._common import ALSBase, TrainContext, TrainingData

_log = logging.getLogger(__name__)


class BiasedMF(ALSBase):
    """
    Biased matrix factorization trained with alternating least squares
    :cite:p:`zhouLargeScaleParallelCollaborative2008`.  This is a
    prediction-oriented algorithm suitable for explicit feedback data, using the
    alternating least squares approach to compute :math:`P` and :math:`Q` to
    minimize the regularized squared reconstruction error of the ratings matrix.

    See the base class :class:`ALSBase` for documentation on the estimated
    parameters you can extract from a trained model.

    Args:
        features:
            The number of features to train.
        epochs:
            The number of iterations to train.
        reg:
            The regularization factor; can also be a tuple ``(ureg, ireg)`` to
            specify separate user and item regularization terms.
        damping:
            Damping term for the bias model.
        rng_spec:
            Random number generator or state (see :func:`seedbank.numpy_rng`).
    """

    timer = None

    features: int
    epochs: int
    reg: float | tuple[float, float]
    damping: UITuple[float]
    rng: np.random.Generator
    save_user_features: bool

    bias_: BiasModel

    def __init__(
        self,
        features: int,
        *,
        epochs: int = 10,
        reg: float | tuple[float, float] = 0.1,
        damping: float | UITuple[float] | tuple[float, float] = 5.0,
        rng_spec: SeedLike | None = None,
        save_user_features: bool = True,
    ):
        super().__init__(
            features,
            epochs=epochs,
            reg=reg,
            rng_spec=rng_spec,
            save_user_features=save_user_features,
        )
        self.damping = UITuple.create(damping)

    @property
    def logger(self):
        return _log

    @override
    def prepare_data(self, data: Dataset):
        # transform ratings using offsets
        rmat = data.interaction_matrix("torch", layout="coo", field="rating")

        _log.info("[%s] normalizing ratings", self.timer)
        self.bias_ = BiasModel.learn(data, self.damping)
        rmat = self.bias_.transform_matrix(rmat)

        rmat = rmat.to_sparse_csr()
        assert not torch.any(torch.isnan(rmat.values()))
        return TrainingData.create(data.users, data.items, rmat)

    @override
    def initial_params(self, nrows: int, ncols: int) -> torch.Tensor:
        mat = self.rng.standard_normal((nrows, ncols))
        mat /= np.linalg.norm(mat, axis=1).reshape((nrows, 1))
        mat = torch.from_numpy(mat)
        return mat

    @override
    def als_half_epoch(self, epoch: int, context: TrainContext):
        chunks = WorkChunks.create(context.nrows)
        with progress_handle(
            _log, f"epoch {epoch} {context.label}s", total=context.nrows, unit="row"
        ) as pbh:
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

        # unpack regularization
        if isinstance(self.reg, tuple):
            ureg, ireg = self.reg
        else:
            ureg = self.reg

        u_feat = _train_bias_row_cholesky(inums[mask], ri_val, self.item_features_, ureg)
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

    def __str__(self):
        return "als.BiasedMF(features={}, regularization={})".format(self.features, self.reg)


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
    A = MMT + regI * len(cols)
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
