from __future__ import annotations

import logging

import numpy as np
from typing_extensions import override

from lenskit.data import Dataset, ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.pipeline import Component, Trainable

_logger = logging.getLogger(__name__)


class TrainingCandidateSelectorBase(Component, Trainable):
    """
    Base class for candidate selectors using the training data.
    """

    items_: Vocabulary

    @property
    def is_trained(self) -> bool:
        return hasattr(self, "items_")

    @override
    def train(self, data: Dataset):
        self.items_ = data.items.copy()


class AllTrainingItemsCandidateSelector(TrainingCandidateSelectorBase):
    """
    Candidate selector that selects all known items from the training data.
    """

    def __call__(self) -> ItemList:
        return ItemList.from_vocabulary(self.items_)


class UnratedTrainingItemsCandidateSelector(TrainingCandidateSelectorBase):
    """
    Candidate selector that selects all known items from the training data that
    do not appear in the request user's history (:attr:`RecQuery.user_items`).
    If no item history is available, then all training items are returned.

    In order to look up the user's history in the training data, this needs to
    be combined with a component like
    :class:`~.history.UserTrainingHistoryLookup`.
    """

    def __call__(self, query: QueryInput) -> ItemList:
        query = RecQuery.create(query)
        items = ItemList.from_vocabulary(self.items_)

        if query.user_items is not None:
            mask = np.full(len(self.items_), True, np.bool_)
            qis = query.user_items.numbers(vocabulary=self.items_)
            qis = qis[qis >= 0]
            mask[qis] = False
            items = items[mask]

        return items
