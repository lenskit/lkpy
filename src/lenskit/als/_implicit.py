# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_array
from typing_extensions import override

from lenskit._accel import als
from lenskit.data import Dataset, ItemList
from lenskit.data.types import NPMatrix, NPVector
from lenskit.math.solve import solve_cholesky

from ._common import ALSBase, ALSConfig, ALSTrainerBase, TrainContext


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

    With weight :math:`w`, this function decomposes the matrix
    :math:`\\mathbb{1}^* + Rw`, where :math:`\\mathbb{1}^*` is an :math:`m
    \\times n` matrix of all 1s.

    See the base class :class:`ALSBase` for documentation on the estimated
    parameters you can extract from a trained model. See
    :class:`ImplicitMFConfig` and :class:`ALSConfig` for the configuration
    options for this component.

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

    config: ImplicitMFConfig

    _OtOr: NPMatrix

    def create_trainer(self, data, options):
        return ImplicitMFTrainer(self, data, options)

    @override
    def new_user_embedding(
        self, user_num: int | None, user_items: ItemList
    ) -> tuple[NPVector, None]:
        ri_idxes = user_items.numbers(vocabulary=self.items)

        ri_good = ri_idxes >= 0
        ri_it = ri_idxes[ri_good]
        if self.config.use_ratings:
            ratings = user_items.field("rating")
            if ratings is None:
                raise ValueError("no ratings in user items")
            ri_val = ratings[ri_good] * self.config.weight
        else:
            ri_val = np.full((len(ri_good),), self.config.weight)

        ri_val = ri_val.astype(self.item_embeddings.dtype)

        u_feat = self._train_new_row(ri_it, ri_val, self.item_embeddings, self._OtOr)
        return u_feat, None

    def _train_new_row(
        self,
        items: NPVector[np.int32],
        ratings: NPVector,
        i_embeds: NPMatrix,
        OtOr: NPMatrix,
    ) -> NPVector:
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


class ImplicitMFTrainer(ALSTrainerBase[ImplicitMFScorer, ImplicitMFConfig]):
    @override
    def prepare_matrix(self, data: Dataset) -> coo_array:
        ints = data.interactions().matrix()
        if self.config.use_ratings:
            rmat = ints.scipy(attribute="rating", layout="coo")
        else:
            rmat = ints.scipy(layout="coo")

        vals = np.require(rmat.data, dtype=np.float32) * self.config.weight
        return coo_array((vals, (rmat.row, rmat.col)), shape=rmat.shape)

    @override
    def initial_params(self, nrows: int, ncols: int) -> NPMatrix:
        mat = self.rng.standard_normal((nrows, ncols), dtype=np.float32) * 0.01
        mat *= mat
        return mat

    @override
    def als_half_epoch(self, epoch: int, context: TrainContext) -> float:
        OtOr = _implicit_otor(context.right, context.reg)
        return als.train_implicit_matrix(context.matrix, context.left, context.right, OtOr)

    @override
    def finalize(self):
        # compute OtOr and save it on the model
        reg = self.config.user_reg
        self.scorer._OtOr = _implicit_otor(self.scorer.item_embeddings, reg)
        super().finalize()


def _implicit_otor(other: NPMatrix, reg: float) -> NPMatrix:
    nf = other.shape[1]
    regmat = np.eye(nf, dtype=other.dtype)
    regmat *= reg
    Ot = other.T
    OtO = Ot @ other
    OtO += regmat
    return OtO
