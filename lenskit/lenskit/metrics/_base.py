from abc import ABC, abstractmethod
from typing import ClassVar, Protocol

from lenskit.data import ItemList, ItemListCollection


class MetricFunction(Protocol):
    "Interface for per-list metrics implemented as simple functions."

    @abstractmethod
    def __call__(self, output: ItemList, test: ItemList, /) -> float: ...


class Metric(ABC):
    """
    Base class for LensKit metrics.  Individual metrics need to implement a
    sub-interface, such as :class:`ListMetric` and/or :class:`GlobalMetric`.

    For simplicity in the analysis code, you cannot simply implement the
    properties of this class on an arbitrary class in order to implement a
    metric with all available behavior such as labeling and defaults; you must
    actually extend this class.  This requirement may be relaxed in the future.
    """

    @property
    def label(self) -> str:
        """
        The metric's default label in output.

        The base implementation returns the class name by default.
        """
        return self.__class__.__name__

    def __str__(self):
        return f"Metric {self.label}"


class ListMetric(Metric):
    """
    Base class for metrics that measure individual recommendation (or
    prediction) lists, and whose results may be aggregated to compute overall
    metrics.

    For prediction metrics, this is *macro-averaging*.
    """

    default: ClassVar[float | None] = 0.0
    """
    The default value to infer when computing statistics over missing values.
    If ``None``, no inference is done (necessary for metrics like RMSE, where
    the missing value is theoretically infinite).
    """

    @abstractmethod
    def measure_list(self, output: ItemList, test: ItemList, /) -> float:
        """
        Compute the metric value for a single result list.

        Individual metric classes need to implement this method.
        """
        raise NotImplementedError()


class GlobalMetric(Metric):
    """
    Base class for metrics that measure entire runs at a time.

    For prediction metrics, this is *micro-averaging*.
    """

    @abstractmethod
    def measure_run(self, output: ItemListCollection, test: ItemListCollection, /) -> float:
        """
        Compute a metric value for an entire run.

        Individual metric classes need to implement this method.
        """
        raise NotImplementedError()
