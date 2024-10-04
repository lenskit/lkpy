import logging

import pandas as pd

from lenskit.data import Dataset, ItemList
from lenskit.pipeline import Component

_log = logging.getLogger(__name__)


class PopScorer(Component):
    """
    Score items by their popularity.  Use with :py:class:`TopN` to get a
    most-popular-items recommender.

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
    item_scores_: pd.Series

    def __init__(self, score_method: str = "quantile"):
        self.score_method = score_method

    def train(self, data: Dataset):
        _log.info("counting item popularity")
        stats = data.item_stats()
        scores = stats["count"]

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

        self.item_scores_ = scores

        return self

    def __call__(self, items: ItemList) -> ItemList:
        scores = self.item_scores_.reindex(items.ids())
        return ItemList(items, scores=scores)

    def __str__(self):
        return "PopScore({})".format(self.score_method)
