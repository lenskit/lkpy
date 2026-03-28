# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
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


class TrainingItemsCandidateSelector(Component[ItemList], Trainable):
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
    items: Vocabulary
    """
    List of known items from the training data.
    """

    @override
    def is_trained(self):
        return hasattr(self, "items")

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        self.items_ = data.items

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
