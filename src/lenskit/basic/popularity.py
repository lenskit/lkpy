# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing_extensions import override

from lenskit.data import Dataset, ItemList, Vocabulary
from lenskit.logging import get_logger
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_log = get_logger(__name__)


class PopConfig(BaseModel):
    """
    Configuration for popularity scoring.
    """

    score: Literal["quantile", "rank", "count"] = "quantile"
    """
    The method for computing popularity scores.  For all methods, higher scores
    represent more popular items.
    """


class PopScorer(Component[ItemList], Trainable):
    """
    Score items by their popularity.  Use with :py:class:`TopN` to get a
    most-popular-items recommender.

    Stability:
        Caller

    Attributes:
        item_pop_:
            Item popularity scores.
    """

    config: PopConfig

    items_: Vocabulary
    item_scores_: np.ndarray[tuple[int], np.dtype[np.float32]]

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        if hasattr(self, "item_scores_") and not options.retrain:
            return

        _log.info("counting item popularity")
        self.items_ = data.items
        stats = data.item_stats()
        scores = stats["count"]
        self.item_scores_ = np.require(self._train_internal(scores).values, np.float32)

    def _train_internal(self, scores: pd.Series) -> pd.Series:
        if self.config.score == "rank":
            _log.info("ranking %d items", len(scores))
            return scores.rank()
        elif self.config.score == "quantile":
            _log.info("computing quantiles for %d items", len(scores))
            cmass = scores.sort_values()
            cmass = cmass.cumsum()
            cdens = cmass / scores.sum()
            return cdens.reindex(scores.index)
        elif self.config.score == "count":
            _log.info("scoring items with their rating counts")
            return scores
        else:
            raise ValueError("invalid scoring method " + repr(self.config.score))

    def __call__(self, items: ItemList) -> ItemList:
        inums = items.numbers(vocabulary=self.items_, missing="negative")
        mask = inums >= 0
        good_inums = inums[mask]
        _log.debug("getting popularity scores", n_good=len(good_inums), n_all=len(inums))
        scores = np.full(len(items), np.nan, np.float32)
        scores[mask] = self.item_scores_[inums[mask]]
        return ItemList(items, scores=scores)


class TimeBoundedPopConfig(PopConfig):
    cutoff: datetime
    """
    Time window for computing popularity scores.
    """


class TimeBoundedPopScore(PopScorer):
    """
    Score items by their time-bounded popularity, i.e., the popularity in the
    most recent `time_window` period.  Use with :py:class:`TopN` to get a
    most-popular-recent-items recommender.

    Attributes:
        item_scores_(pandas.Series):
            Time-bounded item popularity scores.
    """

    config: TimeBoundedPopConfig

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        if hasattr(self, "item_scores_") and not options.retrain:
            return

        _log.info("counting time-bounded item popularity")
        log = data.interaction_table(format="pandas", original_ids=True)

        item_scores = None
        if "timestamp" not in log.columns:
            _log.warning("no timestamps in interaction log; falling back to PopScorer")
            super().train(data, options)
            return
        else:
            item_ids = log["item_id"][log["timestamp"] > self.config.cutoff]
            counts = item_ids.value_counts().reindex(data.items.index, fill_value=0)

            item_scores = super()._train_internal(counts)

        self.items_ = data.items
        self.item_scores_ = np.require(item_scores.reindex(self.items_.ids()).values, np.float32)
