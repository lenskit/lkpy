from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing_extensions import override

from lenskit.data import Dataset, ItemList, Vocabulary
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_log = logging.getLogger(__name__)


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
    item_scores_: np.ndarray[int, np.dtype[np.float32]]

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        if hasattr(self, "item_scores_") and not options.retrain:
            return

        _log.info("counting item popularity")
        self.items_ = data.items.copy()
        stats = data.item_stats()
        scores = stats["count"].reindex(self.items_.ids())
        self.item_scores_ = np.require(
            self._train_internal(scores).reindex(self.items_.ids()).values, np.float32
        )

    def _train_internal(self, scores: pd.Series) -> pd.Series:
        if self.config.score == "rank":
            _log.info("ranking %d items", len(scores))
            scores = scores.rank().sort_index()
        elif self.config.score == "quantile":
            _log.info("computing quantiles for %d items", len(scores))
            cmass = scores.sort_values()
            cmass = cmass.cumsum()
            cdens = cmass / scores.sum()
            scores = cdens.sort_index()
        elif self.config.score == "count":
            _log.info("scoring items with their rating counts")
            scores = scores.sort_index()
        else:
            raise ValueError("invalid scoring method " + repr(self.config.score))

        return scores

    def __call__(self, items: ItemList) -> ItemList:
        inums = items.numbers(vocabulary=self.items_, missing="negative")
        mask = inums >= 0
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
        log = data.interaction_table(format="numpy")

        item_scores = None
        if log.timestamps is None:
            _log.warning("no timestamps in interaction log; falling back to PopScorer")
            super().train(data, options)
            return
        else:
            counts = np.zeros(data.item_count, dtype=np.int32)
            start_timestamp = self.config.cutoff.timestamp()
            item_nums = log.item_nums[log.timestamps > start_timestamp]
            np.add.at(counts, item_nums, 1)

            item_scores = super()._train_internal(
                pd.Series(counts, index=data.items.index),
            )

        self.items_ = data.items.copy()
        self.item_scores_ = np.require(item_scores.reindex(self.items_.ids()).values, np.float32)
