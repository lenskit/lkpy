# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Hierarchical Poisson factorization from ``hpfrec``.
"""

import logging

import hpfrec
import numpy as np
from pydantic import AliasChoices, BaseModel, Field, JsonValue
from typing_extensions import override

from lenskit.data import Dataset, ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_logger = logging.getLogger(__name__)


class HPFConfig(BaseModel, extra="allow"):
    """
    Configuration class for :class:`HPFScorer`.

    Only a couple of configuration options are directly defined here.  Other
    options are taken from the :class:`hpfrec.HPF` constructor's keyword
    arguments.
    """

    __pydantic_extra__: dict[str, JsonValue]
    count_attribute: str | None = "count"
    """
    The attribute from which to get user-item interaction counts.  If ``None``,
    uses indicator variables as a count of 1.
    """

    embedding_size: int = Field(
        default=50, validation_alias=AliasChoices("embedding_size", "features")
    )
    """
    The dimension of user and item embeddings (number of latent features to
    learn).
    """


class HPFScorer(Component[ItemList], Trainable):
    """
    Hierarchical Poisson factorization, provided by `hpfrec`_

    .. _hpfrec: https://hpfrec.readthedocs.io/en/latest/

    Stability:
        Experimental

    Args:
        features:
            the number of features
        kwargs:
            additional arguments to pass to :class:`hpfrec.HPF`.
    """

    config: HPFConfig

    users_: Vocabulary
    user_features_: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    items_: Vocabulary
    item_features_: np.ndarray[tuple[int, int], np.dtype[np.float64]]

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        if hasattr(self, "item_features_") and not options.retrain:
            return

        interacts = data.interactions().matrix()
        matrix = interacts.scipy(self.config.count_attribute, layout="coo")

        hpf = hpfrec.HPF(
            self.config.embedding_size, reindex=False, **self.config.__pydantic_extra__
        )  # type: ignore

        _logger.info("fitting HPF model with %d features", self.config.embedding_size)
        hpf.fit(matrix)

        self.users_ = data.users
        self.items_ = data.items
        self.user_features_ = hpf.Theta  # type: ignore
        self.item_features_ = hpf.Beta  # type: ignore

        return self

    @override
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)

        user_id = query.user_id
        user_num = None
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
