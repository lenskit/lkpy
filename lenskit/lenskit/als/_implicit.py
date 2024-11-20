# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import torch
from seedbank import SeedLike
from typing_extensions import Optional, override

from lenskit.algorithms.als.common import TrainingData
from lenskit.data import Dataset
from lenskit.math.solve import solve_cholesky
from lenskit.parallel.chunking import WorkChunks
from lenskit.util.logging import pbh_update, progress_handle

from ._common import ALSBase, TrainContext

_log = logging.getLogger(__name__)


class ImplicitMF(ALSBase):
    """
    Implicit matrix factorization trained with alternating least squares
    :cite:p:`hu:implicit-mf`.  This algorithm outputs
    'predictions', but they are not on a meaningful scale.  If its input data
    contains ``rating`` values, these will be used as the 'confidence' values;
    otherwise, confidence will be 1 for every rated item.

    See the base class :class:`.MFPredictor` for documentation on the estimated
    parameters you can extract from a trained model.

    With weight :math:`w`, this function decomposes the matrix
    :math:`\\mathbb{1}^* + Rw`, where :math:`\\mathbb{1}^*` is an :math:`m
    \\times n` matrix of all 1s.

    .. versionchanged:: 2025.1
        ``ImplicitMF`` no longer supports multiple training methods. It always uses
        Cholesky decomposition now.

    .. versionchanged:: 0.14
        By default, ``ImplicitMF`` ignores a ``rating`` column if one is present in the training
        data.  This can be changed through the ``use_ratings`` option.

    .. versionchanged:: 0.13
        In versions prior to 0.13, ``ImplicitMF`` used the rating column if it was present.
        In 0.13, we added an option to control whether or not the rating column is used; it
        initially defaulted to ``True``, but with a warning.  In 0.14 it defaults to ``False``.

    Args:
        features:
            The number of features to train
        epochs:
            The number of iterations to train
        reg:
            The regularization factor
        weight:
            The scaling weight for positive samples (:math:`\\alpha` in
            :cite:p:`hu:implicit-mf`).
        use_ratings:
            Whether to use the `rating` column, if present.  Defaults to
            ``False``; when ``True``, the values from the ``rating`` column are
            used, and multipled by ``weight``; if ``False``, ImplicitMF treats
            every rated user-item pair as having a rating of 1.
        rng_spec:
            Random number generator or state (see :func:`~seedbank.numpy_rng`).
        progress: a :func:`tqdm.tqdm`-compatible progress bar function
    """

    weight: float
    use_ratings: bool

    OtOr_: torch.Tensor

    def __init__(
        self,
        features: int,
        *,
        epochs: int = 20,
        reg: float | tuple[float, float] = 0.1,
        weight: float = 40,
        use_ratings: bool = False,
        rng_spec: Optional[SeedLike] = None,
        save_user_features: bool = True,
    ):
        super().__init__(
            features,
            epochs=epochs,
            reg=reg,
            rng_spec=rng_spec,
            save_user_features=save_user_features,
        )
        self.weight = weight
        self.use_ratings = use_ratings

    @property
    def logger(self):
        return _log

    @override
    def fit(self, data: Dataset, **kwargs):
        super().fit(data, **kwargs)

        # compute OtOr and save it on the model
        reg = self.reg[0] if isinstance(self.reg, tuple) else self.reg
        self.OtOr_ = _implicit_otor(self.item_features_, reg)

        return self

    @override
    def prepare_data(self, data: Dataset) -> TrainingData:
        if self.use_ratings:
            rmat = data.interaction_matrix("torch", field="rating")
        else:
            rmat = data.interaction_matrix("torch")

        rmat.values().multiply_(self.weight)
        return TrainingData.create(data.users, data.items, rmat)

    @override
    def initial_params(self, nrows: int, ncols: int) -> torch.Tensor:
        mat = self.rng.standard_normal((nrows, ncols)) * 0.01
        mat = torch.from_numpy(mat)
        mat.square_()
        return mat

    @override
    def als_half_epoch(self, epoch: int, context: TrainContext) -> float:
        chunks = WorkChunks.create(context.nrows)

        OtOr = _implicit_otor(context.right, context.reg)
        with progress_handle(
            _log, f"epoch {epoch} {context.label}s", total=context.nrows, unit="row"
        ) as pbh:
            return _train_implicit_cholesky_fanout(context, OtOr, chunks, pbh)

    @override
    def new_user_embedding(self, user, ratings: pd.Series) -> tuple[torch.Tensor, None]:
        ri_idxes = self.items_.numbers(ratings.index)
        ri_good = ri_idxes >= 0
        ri_it = ri_idxes[ri_good]
        if self.use_ratings:
            ri_val = ratings.values[ri_good] * self.weight
        else:
            ri_val = np.full(len(ri_good), self.weight)

        ri_it = torch.from_numpy(ri_it)
        ri_val = torch.from_numpy(ri_val).type(self.item_features_.dtype)

        u_feat = _train_implicit_row_cholesky(ri_it, ri_val, self.item_features_, self.OtOr_)
        return u_feat, None

    def __str__(self):
        return "als.ImplicitMF(features={}, reg={}, w={})".format(
            self.features, self.reg, self.weight
        )


@torch.jit.script
def _implicit_otor(other: torch.Tensor, reg: float) -> torch.Tensor:
    nf = other.shape[1]
    regmat = torch.eye(nf)
    regmat *= reg
    Ot = other.T
    OtO = Ot @ other
    OtO += regmat
    return OtO


def _train_implicit_row_cholesky(
    items: torch.Tensor, ratings: torch.Tensor, i_embeds: torch.Tensor, OtOr: torch.Tensor
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
    _log.debug("learning new user row with %d items", len(items))

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
