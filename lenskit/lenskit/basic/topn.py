"""
Basic Top-*N* ranking.
"""

import logging

import numpy as np
from pydantic import BaseModel, PositiveInt

from lenskit.data import ItemList
from lenskit.pipeline.components import Component

_log = logging.getLogger(__name__)


class TopNConfig(BaseModel):
    """
    Configuration for top-N ranking.
    """

    n: PositiveInt | None = None
    """
    The number of items to return. -1 or ``None`` to return all scored items.
    """


class TopNRanker(Component):
    """
    Rank scored items by their score and take the top *N*.  The ranking length
    can be passed either at runtime or at component instantiation time, with the
    latter taking precedence.

    Stability:
        Caller
    """

    config: TopNConfig
    "Configuration object."

    def __call__(self, *, items: ItemList, n: int | None = None) -> ItemList:
        """
        Rank the items.

        Args:
            items:
                The items to rank, with scores.  Items with missing scores are
                not included in the final ranking.
            n:
                The number of items to return, or -1 to return all scored items.
                If ``None``, the length configured at construction time is used.

        Returns:
            An ordered list of items, with scores and all other attributes
            preserved.
        """
        if n is None:
            n = self.config.n or -1

        if n >= 0:
            _log.debug("ranking top %d of %d items", n, len(items))
        else:
            _log.debug("ranking all of %d items", len(items))

        scores = items.scores("numpy")
        if scores is None:
            raise RuntimeError("input item list has no scores")

        # find and filter out invalid scores
        v_mask = ~np.isnan(scores)
        items = items[v_mask]

        # order remaining scores
        order = np.argsort(scores[v_mask])
        if n >= 0 and n < len(order):
            order = order[: -(n + 1) : -1]
        else:
            order = order[::-1]

        # now we need to return in expected order
        result = items[order]
        return ItemList(result, ordered=True)
