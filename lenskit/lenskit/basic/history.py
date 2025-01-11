"""
Components that look up user history from the training data.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from typing_extensions import override

from lenskit.data import Dataset, ItemList, QueryInput, RecQuery
from lenskit.data.matrix import CSRStructure
from lenskit.data.vocab import Vocabulary
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_logger = logging.getLogger(__name__)


class UserTrainingHistoryLookup(Component[ItemList], Trainable):
    """
    Look up a user's history from the training data.

    Stability:
        Caller
    """

    training_data_: Dataset

    @override
    def train(self, data: Dataset, options: TrainingOptions):
        # TODO: find a better data structure for this
        if hasattr(self, "training_data_") and not options.retrain:
            return

        self.training_data_ = data

    def __call__(self, query: QueryInput) -> RecQuery:
        """
        Look up the user's data from the training history (if needed), and
        ensure a fully-populated :class:`RecQuery`.
        """
        query = RecQuery.create(query)
        if query.user_id is None:
            return query

        if query.user_items is None:
            query.user_items = self.training_data_.user_row(query.user_id)

        return query

    def __str__(self):
        return self.__class__.__name__


class KnownRatingScorer(Component[ItemList], Trainable):
    """
    Score items by returning their values from the training data.

    Stability:
        Caller

    Args:
        score:
            Whether to score items with their rating values, or a 0/1 indicator
            of their presence in the training data.  The default (``None``) uses
            ratings if available, and otherwise scores with 1 for interacted
            items and leaves non-interacted items unscored.
        source:
            Whether to use the training data or the user's history represented
            in the query as the source of score data.
    """

    score: Literal["rating", "indicator"] | None
    source: Literal["training", "query"]

    users_: Vocabulary
    items_: Vocabulary
    matrix_ = csr_array | CSRStructure

    def __init__(
        self,
        score: Literal["rating", "indicator"] | None = None,
        source: Literal["training", "query"] = "training",
    ):
        self.score = score
        self.source = source

    @override
    def train(self, data: Dataset, options: TrainingOptions):
        if hasattr(self, "matrix_") and not options.retrain:
            return

        if self.source == "query":
            return

        self.users_ = data.users
        self.items_ = data.items
        if self.score == "indicator":
            self.matrix_ = data.interaction_matrix("structure")
        else:
            self.matrix_ = data.interaction_matrix("scipy", field="rating")

    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)

        # figure out what scores we start with
        scores = None
        if self.source == "query" and query.user_items is not None:
            if self.score != "indicator":
                scores = query.user_items.field("rating", "pandas", index="ids")
            if scores is None:
                scores = pd.Series(1.0, index=query.user_items.ids())

        elif (
            self.source == "training" and query.user_id is not None and query.user_id in self.users_
        ):
            urow = self.users_.number(query.user_id)
            if isinstance(self.matrix_, csr_array):
                assert self.score != "indicator"
                # get the user's row as a sparse array
                uarr = self.matrix_[[urow]]
                assert isinstance(uarr, csr_array)
                # create a series
                scores = pd.Series(uarr.data, index=self.items_.ids(uarr.indices))
            elif isinstance(self.matrix_, CSRStructure):
                scores = pd.Series(1.0, index=self.items_.ids(self.matrix_.row_cs(urow)))

        if scores is None:
            scores = pd.Series(np.nan, index=items.ids())

        scores = scores.reindex(
            items.ids(), fill_value=0.0 if self.score == "indicator" else np.nan
        )
        return ItemList(items, scores=scores.values)  # type: ignore

    def __str__(self):
        return self.__class__.__name__
