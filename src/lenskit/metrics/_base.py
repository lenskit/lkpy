# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, Protocol, override

from lenskit.data import ItemList, ItemListCollection
from lenskit.data.accum import (
    Accumulator,
    AccumulatorFactory,
    ValueStatAccumulator,
    ValueStatistics,
)


class MetricFunction(Protocol):
    "Interface for per-list metrics implemented as simple functions."

    @abstractmethod
    def __call__(self, output: ItemList, test: ItemList, /) -> float: ...


class Metric[L, S: float | Mapping[str, float | int | object]](ABC, AccumulatorFactory[L, S]):
    """
    Base class for LensKit metrics.  Individual metrics need to implement a
    sub-interface, such as :class:`ListMetric` and/or :class:`GlobalMetric`.

    This class defines the interface for metrics. Subclasses should implement
    the `measure_list` method to compute metric values.

    The `summarize()` method has a default implementation that computes the mean
    of the per-list metric values, but subclasses can override it to provide
    more appropriate summary statistics.

    .. versionchanged:: 2026.1

        Removed the ``summarize`` method in favor of requiring metrics to
        implement :class:`AccumulatorFactory` to allow metric-controlled
        accumulation.

    Stability:
        Full

    .. note::

        For simplicity in the analysis code, you cannot simply implement the
        properties of this class on an arbitrary class in order to implement a
        metric with all available behavior such as labeling and defaults; you
        must actually extend this class. This requirement may be relaxed in the
        future.

    The default value to impute when computing statistics over missing values.
    If ``None``, no imputation is done (necessary for metrics like RMSE, where
    the missing value is theoretically infinite).
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

    @abstractmethod
    def measure_list(self, output: ItemList, test: ItemList, /) -> L:
        """
        Compute measurements for a single list.

        Returns:
            - A float for simple metrics
            - Intermediate data for decomposed metrics
            - A dict mapping metric names to values for multi-metric classes
        """
        raise NotImplementedError()  # pragma: no cover

    def extract_list_metrics(self, data: L, /) -> float | dict[str, float] | None:
        """
        Extract per-list metric(s) from intermediate measurement data.

        Returns:
            - A float for simple metrics
            - A dict mapping metric names to values for multi-metric classes
            - None if no per-list metrics are available
        """
        return None

    @abstractmethod
    def create_accumulator(self) -> Accumulator[L, S]:  # pragma: nocov
        """
        Creaet an accumulator to aggregate per-list measurements into summary
        metrics.

        Each result from :meth:`measure_list` is passed to
        :meth:`Accumulator.add`.
        """
        raise NotImplementedError()


class ListMetric(Metric[float, ValueStatistics]):
    """
    Base class for metrics defined on individual recommendation outputs.  This
    is the most common type of metric.

    For prediction metrics, this is *macro-averaging*.

    Metrics based on this class implement :meth:`measure_list` to compute a
    single numeric value for each list, and the accumulated result will be basic
    statistical summaries of those values.

    Stability:
        Full
    """

    default: ClassVar[float | None] = 0.0

    @abstractmethod
    def measure_list(self, output: ItemList, test: ItemList, /) -> float: ...

    @override
    def extract_list_metrics(self, data: Any, /) -> float:
        """
        Return the given per-list metric result.
        """
        return data

    @override
    def create_accumulator(self) -> ValueStatAccumulator:
        return ValueStatAccumulator()


class GlobalMetric:
    """
    Base class for metrics that measure entire runs at a time.

    For prediction metrics, this is *micro-averaging*.

    Stability:
        Full
    """

    @abstractmethod
    def measure_run(self, output: ItemListCollection, test: ItemListCollection, /) -> float:
        """
        Compute a metric value for an entire run.

        Individual metric classes need to implement this method.
        """
        raise NotImplementedError()  # pragma: no cover
