# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging

import hpfrec
import numpy as np
from typing_extensions import Any, override

from lenskit.data import Dataset, ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.pipeline import Component, Trainable

_logger = logging.getLogger(__name__)


class HPFScorer(Component, Trainable):
    """
    Hierarchical Poisson factorization, provided by
    `hpfrec <https://hpfrec.readthedocs.io/en/latest/>`_.

    .. todo::
        Right now, this uses the 'rating' as a count. Actually use counts.

    Args:
        features:
            the number of features
        kwargs:
            additional arguments to pass to :py:class:`hpfrec.HPF`.
    """

    features: int
    _kwargs: dict[str, Any]

    users_: Vocabulary
    user_features_: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    items_: Vocabulary
    item_features_: np.ndarray[tuple[int, int], np.dtype[np.float64]]

    def __init__(self, features: int = 50, **kwargs):
        self.features = features
        self._kwargs = kwargs

    def get_config(self):
        return {"features": self.features} | self._kwargs

    @property
    def is_trained(self) -> bool:
        return hasattr(self, "item_features_")

    @override
    def train(self, data: Dataset):
        log = data.interaction_matrix("pandas", field="rating")
        log = log.rename(
            columns={
                "user_num": "UserId",
                "item_num": "ItemId",
                "rating": "Count",
            }
        )

        hpf = hpfrec.HPF(self.features, reindex=False, **self._kwargs)

        _logger.info("fitting HPF model with %d features", self.features)
        hpf.fit(log)

        self.users_ = data.users
        self.items_ = data.items
        self.user_features_ = hpf.Theta  # type: ignore
        self.item_features_ = hpf.Beta  # type: ignore

        return self

    @override
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)

        user_id = query.user_id
        if user_id is not None:
            user_num = self.users_.number(user_id, missing=None)
        if user_num is None:
            _logger.debug("unknown user %s", query.user_id)
            return ItemList(items, scores=np.nan)

        u_feat = self.user_features_[user_num, :]

        item_nums = items.numbers(vocabulary=self.items_, missing="negative")
        item_mask = item_nums >= 0
        i_feats = self.item_features_[item_nums[item_mask], :]

        scores = np.full((len(items),), np.nan, dtype=np.float64)
        scores[item_mask] = i_feats @ u_feat

        return ItemList(items, scores=scores)
