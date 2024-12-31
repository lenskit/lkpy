"""
Basic set statistics.
"""

from lenskit.data.items import ItemList

from ._base import Metric


class ListLength(Metric):
    """
    Report the length of the output (recommendation list or predictions).

    Stability:
        Caller
    """

    label = "N"  # type: ignore

    def __call__(self, recs: ItemList, test: ItemList) -> float:
        return len(recs)


class TestItemCount(Metric):
    """
    Report the number of test items.

    Stability:
        Caller
    """

    label = "TestItemCount"  # type: ignore

    def __call__(self, recs: ItemList, test: ItemList) -> float:
        return len(test)
