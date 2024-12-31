from lenskit.data import ItemList

from .._base import GlobalMetric, ListMetric, Metric

__all__ = ["Metric", "ListMetric", "GlobalMetric", "RankingMetricBase"]


class RankingMetricBase(Metric):
    """
    Base class for most ranking metrics, implementing a ``k`` parameter for
    truncation.

    Args:
        k:
            Specify the length cutoff for rankings. Rankings longer than this
            will be truncated prior to measurement.

    Stability:
        Caller
    """

    k: int | None = None
    "The maximum length of rankings to consider."

    def __init__(self, k: int | None = None):
        if k is not None and k < 0:
            raise ValueError("k must be positive or None")
        self.k = k

    def truncate(self, items: ItemList):
        """
        Truncate an item list if it is longer than :attr:`k`.
        """
        if self.k is not None:
            if not items.ordered:
                raise ValueError("top-k filtering requires ordered list")
            if len(items) > self.k:
                return items[: self.k]

        return items
