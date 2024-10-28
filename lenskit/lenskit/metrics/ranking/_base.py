from abc import abstractmethod
from typing import Protocol

from lenskit.data import ItemList


class RankingMetric(Protocol):
    """
    Base class / protocol for ranking metrics.
    """

    @abstractmethod
    def __call__(self, recs: ItemList, test: ItemList) -> float: ...
