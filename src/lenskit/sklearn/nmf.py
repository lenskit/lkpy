# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Nonnegative matrix factorization for implicit feedback.

This module contains a non-negative factorization implicit-feedback scorer built on
:func:`sklearn.decomposition.non_negative_factorization`.
"""

from __future__ import annotations

import numpy as np
from pydantic import AliasChoices, BaseModel, Field, PositiveInt
from sklearn.decomposition import non_negative_factorization
from typing_extensions import Literal, override

from lenskit.config.common import EmbeddingSizeMixin
from lenskit.data import Dataset, ItemList, QueryInput, RecQuery
from lenskit.data.types import NPMatrix
from lenskit.data.vocab import Vocabulary
from lenskit.logging import Stopwatch, get_logger
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_log = get_logger(__name__)


class NMFConfig(EmbeddingSizeMixin, BaseModel, extra="forbid"):
    """
    Configuration for :class:`NMFScorer`.
    See the documentation for :func:`sklearn.decomposition.non_negative_factorization`
    for the configuration options.

    """

    beta_loss: Literal["frobenius", "kullback-leibler", "itakura-saito"] = "frobenius"
    max_iter: PositiveInt = Field(default=200, validation_alias=AliasChoices("max_iter", "epochs"))
    embedding_size: PositiveInt | None = Field(
        default=None, validation_alias=AliasChoices("embedding_size", "n_components")
    )
    alpha_W: float = 0.0
    alpha_H: float | Literal["same"] = "same"
    l1_ratio: float = 0.0


class NMFScorer(Component[ItemList], Trainable):
    """
    Non-negative matrix factorization for implicit feedback using SciKit-Learn's
    :func:`sklearn.decomposition.non_negative_factorization`. It computes the user
    and item embedding matrices using an indicator matrix as the input.

    Stability:
        Caller
    """

    config: NMFConfig

    users: Vocabulary
    items: Vocabulary
    user_components: NPMatrix
    item_components: NPMatrix

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        if hasattr(self, "item_components") and not options.retrain:
            return

        timer = Stopwatch()

        _log.info("[%s] sparsifying and normalizing matrix", timer)
        r_mat = data.interactions().matrix().scipy(layout="csr", legacy=True)

        _log.info("[%s] training NMF", timer)
        W, H, n_iter = non_negative_factorization(
            r_mat,
            beta_loss=self.config.beta_loss,
            max_iter=self.config.max_iter,
            n_components=self.config.embedding_size,
            alpha_W=self.config.alpha_W,
            alpha_H=self.config.alpha_H,
            l1_ratio=self.config.l1_ratio,
        )
        _log.info("[%s] Trained NMF in %d iterations", timer, n_iter)

        self.user_components = np.require(W, dtype=np.float32)
        self.item_components = np.require(H.T, dtype=np.float32)
        self.users = data.users
        self.items = data.items
        _log.info("finished model training in %s", timer)

    @override
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)

        uidx = None
        if query.user_id is not None:
            uidx = self.users.number(query.user_id, missing="none")

        if uidx is None:
            return ItemList(items, scores=np.nan)

        # Get index for user & usable items
        iidx = items.numbers(vocabulary=self.items, missing="negative")
        good_iidx = iidx[iidx >= 0]

        _log.debug("reverse-transforming user %s (idx=%d)", query.user_id, uidx)
        W = self.user_components[[uidx], :]
        H = self.item_components[good_iidx, :]
        S = W @ H.T

        scores = np.full(len(items), np.nan)
        scores[iidx >= 0] = S[0, :]

        return ItemList(items, scores=scores)
