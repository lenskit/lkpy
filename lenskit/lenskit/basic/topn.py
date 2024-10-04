"""
Basic Top-*N* ranking.
"""

import numpy as np

from lenskit.data import ItemList
from lenskit.pipeline import Component


class TopN(Component):
    """
    Rank scored items by their score and take the top *N*.  The ranking length
    can be passed either at runtime or at component instantiation time, with the
    latter taking precedence.

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
        return result
