# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LightFM integration with LensKit.
"""
import sys
from lightfm import LightFM
import numpy as np
from pydantic import BaseModel, JsonValue
from typing import Optional
from typing_extensions import override
from lenskit.data import Dataset, ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

logging = importlib.import_module("logging")
_logger = logging.getLogger(__name__)

class LightFMConfig(BaseModel, extra="allow"):
    __pydantic_extra__: dict[str, JsonValue]
    no_components: int = 50
    """
    Number of latent factors to use.
    """
    learning_rate: float = 0.05
    """
    Learning rate for the LightFM model.
    """
    loss: str = "warp"
    """
    Loss function to optimize. Options: 'logistic', 'bpr', 'warp', or 'warp-kos'.
    """
    epochs: int = 30
    """
    Number of training epochs.
    """
    field: Optional[str] = None
    """
    Field to use for the interaction matrix. Use None for implicit feedback (binary matrix)
    or 'rating' for explicit feedback. Defaults to None.
    """

class LightFMScorer(Component[ItemList], Trainable):
    """
    LightFM scorer for LensKit.

    This uses the LightFM library to implement a hybrid collaborative filtering recommender.

    Stability:
        Caller

    Args:
        config:
            Configuration for the LightFM model.
        kwargs:
            Additional arguments to pass to the LightFM constructor.
    """

    config: LightFMConfig

    users: Vocabulary
    items: Vocabulary
    model: LightFM

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        if hasattr(self, "model") and not options.retrain:
            return

        # Use configurable field for interaction matrix
        interaction_matrix = data.interaction_matrix(format="csr", field=self.config.field)

        _logger.info("Initializing LightFM model with %d components and loss='%s'",
                     self.config.no_components, self.config.loss)
        self.model = LightFM(
            no_components=self.config.no_components,
            learning_rate=self.config.learning_rate,
            loss=self.config.loss,
            **self.config.__pydantic_extra__
        )

        _logger.info("Fitting LightFM model with %d epochs", self.config.epochs)
        self.model.fit(interaction_matrix, epochs=self.config.epochs, num_threads=options.threads)

        self.users = data.users
        self.items = data.items

    @override
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)

        user_id = query.user_id
        user_num = None
        if user_id is not None:
            user_num = self.users.number(user_id, missing=None)
        if user_num is None:
            _logger.debug("Unknown user %s", query.user_id)
            return ItemList(items, scores=np.nan)

        item_nums = items.numbers(vocabulary=self.items, missing="negative")
        item_mask = item_nums >= 0

        scores = np.full((len(items),), np.nan, dtype=np.float64)
        if np.any(item_mask):
            _logger.debug("Scoring user %s on %d items", query.user_id, np.sum(item_mask))
            scores[item_mask] = self.model.predict(
                user_ids=user_num,
                item_ids=item_nums[item_mask],
                num_threads=1
            )

        return ItemList(items, scores=scores)