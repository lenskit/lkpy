from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from typing_extensions import override

from lenskit.data import Dataset, ItemList, Vocabulary
from lenskit.pipeline import Component, Trainable

_log = logging.getLogger(__name__)


class PopScorer(Component, Trainable):
    """
    Score items by their popularity.  Use with :py:class:`TopN` to get a
    most-popular-items recommender.

    Stability:
        Caller

    Args:
        score_type:
            The method for computing popularity scores.  Can be one of the following:

            - ``'quantile'`` (the default)
            - ``'rank'``
            - ``'count'``

    Attributes:
        item_pop_:
            Item popularity scores.
    """

    score_method: str

    items_: Vocabulary
    item_scores_: np.ndarray[int, np.dtype[np.float32]]

    def __init__(self, score_method: str = "quantile"):
        self.score_method = score_method

    @property
    def is_trained(self) -> bool:
        return hasattr(self, "item_scores_")

    @override
    def train(self, data: Dataset):
        _log.info("counting item popularity")
        self.items_ = data.items.copy()
        stats = data.item_stats()
        scores = stats["count"].reindex(self.items_.ids())
        self.item_scores_ = np.require(
            self._train_internal(scores).reindex(self.items_.ids()).values, np.float32
        )

        return self

    def _train_internal(self, scores: pd.Series):
        if self.score_method == "rank":
            _log.info("ranking %d items", len(scores))
            scores = scores.rank().sort_index()
        elif self.score_method == "quantile":
            _log.info("computing quantiles for %d items", len(scores))
            cmass = scores.sort_values()
            cmass = cmass.cumsum()
            cdens = cmass / scores.sum()
            scores = cdens.sort_index()
        elif self.score_method == "count":
            _log.info("scoring items with their rating counts")
            scores = scores.sort_index()
        else:
            raise ValueError("invalid scoring method " + repr(self.score_method))

        return scores

    def __call__(self, items: ItemList) -> ItemList:
        inums = items.numbers(vocabulary=self.items_, missing="negative")
        mask = inums >= 0
        scores = np.full(len(items), np.nan, np.float32)
        scores[mask] = self.item_scores_[inums[mask]]
        return ItemList(items, scores=scores)

    def __str__(self):
        return "PopScore({})".format(self.score_method)


class TimeBoundedPopScore(PopScorer):
    """
    Score items by their time-bounded popularity, i.e., the popularity in the
    most recent `time_window` period.  Use with :py:class:`TopN` to get a
    most-popular-recent-items recommender.

    Args:
        time_window(datetime.timedelta):
            The time window for computing popularity scores.
        score_type(str):
            The method for computing popularity scores.  Can be one of the following:

            - ``'quantile'`` (the default)
            - ``'rank'``
            - ``'count'``

    Attributes:
        item_scores_(pandas.Series):
            Time-bounded item popularity scores.
    """

    def __init__(self, cutoff: datetime, score_method="quantile"):
        super().__init__(score_method)

        self.cutoff = cutoff
        self.score_method = score_method

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
            start_timestamp = self.cutoff.timestamp()
            item_nums = log.item_nums[log.timestamps > start_timestamp]
            np.add.at(counts, item_nums, 1)

            item_scores = super()._train_internal(
                pd.Series(counts, index=data.items.index),
                **kwargs,
            )

        self.item_scores_ = item_scores

        return self

    @override
    def __str__(self):
        return "TimeBoundedPopScore({}, {})".format(self.cutoff, self.score_method)
