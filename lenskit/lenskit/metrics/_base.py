from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from lenskit.data import ItemList


class Metric(Protocol):
    """
    Core protocol implemented by LensKit metrics.
    """

    @abstractmethod
    def __call__(self, output: ItemList, test: ItemList, /) -> float: ...


@runtime_checkable
class LabeledMetric(Metric, Protocol):
    label: str
    """
    The metric's label.
    """
    mean_label: str
    """
    The label to use for the mean of the metric.
    """


class MetricBase(ABC, LabeledMetric):
    """
    Base class for LensKit metric classes, supporting configuration options and
    labels.
    """

    @property
    def label(self):
        """
        The metric's default label in output.

        The base class implementation returns the class name by default.
        """
        return self.__name__

    @property
    def mean_label(self):
        """
        The label to use when aggregating the metric by taking the mean.

        The base class implementation delegates to :attr:`label`.
        """
        return self.label
