from abc import ABC, abstractmethod
from typing import ClassVar, Protocol

from lenskit.data import ItemList


class MetricFunction(Protocol):
    "Interface for metrics implemented as simple functions."

    @abstractmethod
    def __call__(self, output: ItemList, test: ItemList, /) -> float: ...


class Metric(ABC):
    """
    Base class for LensKit metrics.

    For simplicity in the analysis code, you cannot simply implement the
    properties of this class on an arbitrary class in order to implement a
    metric with all available behavior such as labeling and defaults; you must
    actually extend this class.  This requirement may be relaxed in the future.
    """

    default: ClassVar[float | None] = 0.0
    """
    The default value to infer when computing statistics over missing values.
    If ``None``, no inference is done (necessary for metrics like RMSE, where
    the missing value is theoretically infinite).
    """

    @property
    def label(self) -> str:
        """
        The metric's default label in output.

        The base implementation returns the class name by default.
        """
        return self.__class__.__name__

    @property
    def mean_label(self) -> str:
        """
        The label to use when aggregating the metric by taking the mean.

        The base implementation delegates to :attr:`label`.
        """
        return self.label

    @abstractmethod
    def __call__(self, output: ItemList, test: ItemList, /) -> float:
        """
        Metrics should be callable to compute their values.
        """
        ...

    def __str__(self):
        return f"Metric {self.label}"
