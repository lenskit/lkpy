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
from pydantic import AliasChoices, Field
from sklearn.decomposition import non_negative_factorization
from typing_extensions import Literal, override

from lenskit.data import Dataset, ItemList, QueryInput, RecQuery
from lenskit.data.vocab import Vocabulary
from lenskit.logging import Stopwatch, get_logger
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_log = get_logger(__name__)

class NMFScorer(Component[ItemList], Trainable):
    
    users_: Vocabulary
    items_: Vocabulary
    user_components_: NDArray[np.float64]

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        
        r_mat = data.interaction_matrix(format="scipy", layout="coo", legacy=True)
        r_mat = r_mat.tocsr()
        W, H, n_iter = non_negative_factorization(r_mat)

        self.user_components_ = np.dot(W, H)
        
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
        Xt = self.user_components_[[uidx], :]
        X = np.linalg.inv(Xt)
        # restrict to usable desired items
        Xsel = X[0, good_iidx]

        scores = np.full(len(items), np.nan)
        scores[iidx >= 0] = Xsel

        return ItemList(items, scores=scores)