# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
This is for implicit 

This module contains a truncated SVD explicit-feedback scorer built on
:class:`sklearn.decomposition.TruncatedSVD`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import non_negative_factorization
from typing_extensions import Literal, override, Union

from lenskit.data import Dataset, ItemList, QueryInput, RecQuery
from lenskit.data.vocab import Vocabulary
from lenskit.logging import Stopwatch, get_logger
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_log = get_logger(__name__)

@dataclass
class NMFConfig:

    beta_loss: Literal["frobenius", "kullback-leibler", "itakura-saito"] = "frobenius"
    max_iter: int = 200
    n_components: Union[int, None] = None
    alpha_W: float = 0.0
    alpha_H: Union[float, Literal["same"]] = "same"
    l1_ratio: float = 0.0


class NMFScorer(Component[ItemList], Trainable):

    config: NMFConfig
    
    users_: Vocabulary
    items_: Vocabulary
    user_components_: NDArray[np.float64]
    item_components_: NDArray[np.float64]

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        if hasattr(self, "item_components_") and not options.retrain:
            return
        
        timer = Stopwatch()

        _log.info("[%s] sparsifying and normalizing matrix", timer)
        r_mat = data.interaction_matrix(format="scipy", layout="coo", legacy=True)

        r_mat = r_mat.tocsr()

        W, H, n_iter = non_negative_factorization(
            r_mat, beta_loss=self.config.beta_loss, max_iter=self.config.max_iter,
            n_components=self.config.n_components,alpha_W=self.config.alpha_W,
            alpha_H=self.config.alpha_H, l1_ratio=self.config.l1_ratio
        )
        _log.info("[%s] training NMF", timer)

        self.user_components_ = W
        self.item_components_ = H.T
        self.users_ = data.users
        self.items_ = data.items
        _log.info("finished model training in %s", timer)

    @override
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)

        uidx = None
        if query.user_id is not None:
            uidx = self.users_.number(query.user_id, missing="none")

        if uidx is None:
            return ItemList(items, scores=np.nan)

        # Get index for user & usable items
        iidx = items.numbers(vocabulary=self.items_, missing="negative")
        good_iidx = iidx[iidx >= 0]

        _log.debug("reverse-transforming user %s (idx=%d)", query.user_id, uidx)
        W = self.user_components_[[uidx], :]
        H = self.item_components_[good_iidx, :]
        S = W @ H.T

        scores = np.full(len(items), np.nan)
        scores[iidx >= 0] = S[0, :]

        return ItemList(items, scores=scores)