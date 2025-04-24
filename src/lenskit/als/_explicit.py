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
from lenskit.basic import BiasModel, Damping
from lenskit.data import Dataset, ItemList
from lenskit.data.types import NPMatrix, NPVector
from lenskit.math.solve import solve_cholesky

from ._common import ALSBase, ALSConfig, ALSTrainerBase, TrainContext


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
    bias: BiasModel

    def create_trainer(self, data, options):
        return BiasedMFTrainer(self, data, options)

    @override
    def new_user_embedding(
        self, user_num: int | None, items: ItemList
    ) -> tuple[NPVector, float | None]:
        inums = items.numbers(vocabulary=self.items_, missing="negative")
        mask = inums >= 0
        ratings = items.field("rating")
        assert ratings is not None

        biases, u_bias = self.bias.compute_for_items(items, None, items)
        ratings = ratings - biases

        ri_val = ratings[mask]

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
            if user_num is not None and self.bias.user_biases is not None:
                user_bias = self.bias.user_biases[user_num]
            else:
                user_bias = 0.0
        assert user_bias is not None

        biases = self.bias.compute_for_items(items, bias=user_bias)
        scores = scores + biases

        return ItemList(items, scores=scores)


class BiasedMFTrainer(ALSTrainerBase[BiasedMFScorer, BiasedMFConfig]):
    @override
    def prepare_matrix(self, data: Dataset) -> coo_array:
        # transform ratings using offsets
        rmat = data.interactions().matrix().scipy(attribute="rating", layout="coo")

        self.logger.info("normalizing ratings")
        self.scorer.bias = BiasModel.learn(data, damping=self.config.damping)
        rmat = self.scorer.bias.transform_matrix(rmat)

        return rmat.astype(np.float32)

    @override
    def initial_params(self, nrows: int, ncols: int) -> NPMatrix:
        mat = self.rng.standard_normal((nrows, ncols), dtype=np.float32)
        mat /= np.linalg.norm(mat, axis=1).reshape((nrows, 1))
        return mat

    @override
    def als_half_epoch(self, epoch: int, context: TrainContext) -> float:
        return als.train_explicit_matrix_cd(
            context.matrix, context.left, context.right, context.reg
        )


def _train_bias_row_cholesky(
    items: NPVector[np.int32], ratings: NPVector, other: NPMatrix, reg: float
) -> NPVector:
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
        return np.zeros(nf, dtype=np.float32)

    M = other[items, :]
    regI = np.eye(nf, dtype=np.float32) * reg
    MMT = M.T @ M
    A = MMT + regI * len(items)

    V = M.T @ ratings
    x = solve_cholesky(A, V.astype(A.dtype))

    return np.require(x, dtype=np.float32)
