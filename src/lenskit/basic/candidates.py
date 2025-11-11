# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from typing_extensions import override

from lenskit.data import Dataset, ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_logger = logging.getLogger(__name__)


class TrainingCandidateSelectorBase(Component[ItemList], Trainable):
    """
    Base class for candidate selectors using the training data.

    Stability:
        Caller
    """

    config: None
    items_: Vocabulary
    """
    List of known items from the training data.
    """

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        if hasattr(self, "items_") and not options.retrain:
            return

        self.items_ = data.items


class AllTrainingItemsCandidateSelector(TrainingCandidateSelectorBase):
    """
    Candidate selector that selects all known items from the training data.

    Stability:
        Caller
    """

    config: None

    def __call__(self) -> ItemList:
        return ItemList.from_vocabulary(self.items_)


class UnratedTrainingItemsCandidateSelector(TrainingCandidateSelectorBase):
    """
    Candidate selector that selects all known items from the training data that
    do not appear in the request user's history (:attr:`RecQuery.history_items`).
    If no item history is available, then all training items are returned.

    In order to look up the user's history in the training data, this needs to
    be combined with a component like
    :class:`~.history.UserTrainingHistoryLookup`.

    Stability:
        Caller
    """

    config: None

    def __call__(self, query: QueryInput) -> ItemList:
        query = RecQuery.create(query)
        items = ItemList.from_vocabulary(self.items_)

        if query.user_items is not None:
            items = items.remove(numbers=query.user_items.numbers(vocabulary=self.items_))

        return items


@dataclass
class TrainingItemsCandidateConfig:
    """
    Configuration for :class:`TrainingItemsCandidateSelector`.
    """

    exclude: (
        None
        | Literal["query", "all", "history", "session", "context"]
        | Sequence[Literal["history", "session", "context"]]
    ) = "query"
    """
    Specify the items from the query to exclude from the candidates.

    ``None``
        Exclude no items from the request.

    ``"query"``
        Excludes the *query items* (:attr:`RecQuery.query_items`) from the request.
        This is the default.

    ``"all"``
        Exclude all mentioned items from the request.

    ``"history"``, ``"session"``, ``"context"``
        Exclude the specified items from the request.
    """


class TrainingItemsCandidateSelector(TrainingCandidateSelectorBase):
    """
    Candidate selector that selects all known items from the training data, optionally
    excluding certain items from the query (i.e., the request user's history).

    In order to look up the user's history in the training data, this needs to
    be combined with a component like
    :class:`~.history.UserTrainingHistoryLookup`.

    Stability:
        Caller
    """

    config: TrainingItemsCandidateConfig

    def __call__(self, query: QueryInput) -> ItemList:
        query = RecQuery.create(query)
        items = ItemList.from_vocabulary(self.items_)

        exclude = None
        if self.config.exclude == "all":
            exclude = query.all_items
        elif self.config.exclude == "query":
            exclude = query.query_items
        elif isinstance(self.config.exclude, str):
            exclude = query.combined_items(self.config.exclude)
        elif self.config.exclude is not None:
            exclude = query.combined_items(*self.config.exclude)

        if exclude is not None:
            items = items.remove(numbers=exclude.numbers(vocabulary=self.items_))

        return items
