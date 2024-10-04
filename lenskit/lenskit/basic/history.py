"""
Components that look up user history from the training data.
"""

from __future__ import annotations

import logging

from typing_extensions import override

from lenskit.data import Dataset
from lenskit.data.query import QueryInput, RecQuery
from lenskit.pipeline import Component, Trainable

_logger = logging.getLogger(__name__)


class UserTrainingHistoryLookup(Component, Trainable):
    """
    Look up a user's history from the training data.
    """

    @override
    def train(self, data: Dataset):
        # TODO: find a better data structure for this
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
