"""
Basic Top-*N* ranking.
"""

import logging

from lenskit.data import ItemList
from lenskit.pipeline.components import Component
from lenskit.stats import argtopn

_log = logging.getLogger(__name__)


class TopNRanker(Component):
    """
    Rank scored items by their score and take the top *N*.  The ranking length
    can be passed either at runtime or at component instantiation time, with the
    latter taking precedence.

    Stability:
        Caller

    Args:
        n:
            The desired ranking length.  If negative, then scored items are
            ranked but the ranking is not truncated.
    """

    n: int

    def __init__(self, n: int = -1):
        self.n = n

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
            n = self.n

        if n >= 0:
            _log.debug("ranking top %d of %d items", n, len(items))
        else:
            _log.debug("ranking all of %d items", len(items))

        scores = items.scores("numpy")
        if scores is None:
            raise RuntimeError("input item list has no scores")

        order = argtopn(scores, n)

        # now we need to return in expected order
        result = items[order]
        return ItemList(result, ordered=True)
