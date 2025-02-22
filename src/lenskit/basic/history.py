# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Components that look up user history from the training data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from typing_extensions import override

from lenskit.data import Dataset, ItemList, QueryInput, RecQuery
from lenskit.data.dataset import MatrixRelationshipSet
from lenskit.diagnostics import DataError
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_logger = logging.getLogger(__name__)


@dataclass
class LookupConfig:
    interaction_class: str | None = None
    """
    The name of the interaction class to use.  Leave ``None`` to use the
    dataset's default interaction class.
    """


class UserTrainingHistoryLookup(Component[ItemList], Trainable):
    """
    Look up a user's history from the training data.

    Stability:
        Caller
    """

    config: LookupConfig

    interactions: MatrixRelationshipSet

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        # TODO: find a better data structure for this
        if hasattr(self, "interactions") and not options.retrain:
            return

        self.interactions = data.interactions(self.config.interaction_class).matrix()
        if self.interactions.row_type != "user":  # pragma: nocover
            raise DataError("interactions must have user rows")
        if self.interactions.col_type != "item":  # pragma: nocover
            raise DataError("interactions must have item columns")

    def __call__(self, query: QueryInput) -> RecQuery:
        """
        Look up the user's data from the training history (if needed), and
        ensure a fully-populated :class:`RecQuery`.
        """
        query = RecQuery.create(query)
        if query.user_id is None:
            return query

        if query.user_items is None:
            query.user_items = self.interactions.row_items(query.user_id)

        return query


@dataclass
class KnownRatingConfig(LookupConfig):
    score: Literal["rating", "indicator"] | None = None
    """
    The field name to use to score items, or ``"indicator"`` to score with 0/1
    based on presence in the training data.  The default, ``None``, uses ratings
    if available, and otherwise scores with ` for interacted items and leaves
    un-interacted items unscored.
    """
    source: Literal["training", "query"] = "training"
    """
    Whether to get the known ratings from the training data or from the query.
    """


class KnownRatingScorer(Component[ItemList], Trainable):
    """
    Score items by returning their values from the training data.

    Stability:
        Caller
    """

    config: KnownRatingConfig
    interactions: MatrixRelationshipSet

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        if hasattr(self, "interactions") and not options.retrain:
            return

        if self.config.source == "query":
            return

        self.interactions = data.interactions(self.config.interaction_class).matrix()

    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)

        # figure out what scores we start with
        ilist = None
        if self.config.source == "query" and query.user_items is not None:
            ilist = query.user_items

        elif (
            self.config.source == "training"
            and query.user_id is not None
            and query.user_id in self.interactions.row_vocabulary
        ):
            ilist = self.interactions.row_items(query.user_id)

        if ilist is None:
            scores = None
        elif self.config.score == "indicator":
            scores = pd.Series(1.0, index=ilist.ids())
        else:
            scores = ilist.field("rating", format="pandas", index="ids")
            if scores is None and self.config.score is None:
                scores = pd.Series(1.0, index=ilist.ids())

        if scores is None:
            scores = pd.Series(np.nan, index=items.ids())

        scores = scores.reindex(
            items.ids(), fill_value=0.0 if self.config.score == "indicator" else np.nan
        )
        return ItemList(items, scores=scores.values)  # type: ignore
