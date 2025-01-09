import logging
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing_extensions import override

from lenskit.data import Dataset, ItemList
from lenskit.pipeline import Component, Trainable

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


class PopScorer(Component, Trainable):
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

    item_scores_: pd.Series

    @property
    def is_trained(self) -> bool:
        return hasattr(self, "item_scores_")

    @override
    def train(self, data: Dataset):
        _log.info("counting item popularity")
        stats = data.item_stats()
        scores = stats["count"]
        self.item_scores_ = self._train_internal(scores)

        return self

    def _train_internal(self, scores: pd.Series):
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
        scores = self.item_scores_.reindex(items.ids())
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
    def train(self, data: Dataset, **kwargs):
        _log.info("counting time-bounded item popularity")

        log = data.interaction_log("numpy")

        item_scores = None
        if log.timestamps is None:
            _log.warning("no timestamps in interaction log; falling back to PopScorer")
            item_scores = super().train(data, **kwargs).item_scores_
        else:
            counts = np.zeros(data.item_count, dtype=np.int32)
            start_timestamp = self.config.cutoff.timestamp()
            item_nums = log.item_nums[log.timestamps > start_timestamp]
            np.add.at(counts, item_nums, 1)

            item_scores = super()._train_internal(
                pd.Series(counts, index=data.items.index),
                **kwargs,
            )

        self.item_scores_ = item_scores

        return self
