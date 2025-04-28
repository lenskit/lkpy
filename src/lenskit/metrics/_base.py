# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

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

    Stability:
        Full
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

    Stability:
        Full
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

    Stability:
        Full
    """

    @abstractmethod
    def measure_run(self, output: ItemListCollection, test: ItemListCollection, /) -> float:
        """
        Compute a metric value for an entire run.

        Individual metric classes need to implement this method.
        """
        raise NotImplementedError()


class DecomposedMetric(Metric):
    """
    Base class for metrics that measure entire runs through flexible
    aggregations of per-list intermediate measurements.  They can optionally
    extract individual-list metrics from the per-list measurements.

    Stability:
        Full
    """

    @abstractmethod
    def compute_list_data(self, output: ItemList, test: ItemList, /) -> object:
        """
        Compute measurements for a single list.
        """
        raise NotImplementedError()

    @abstractmethod
    def extract_list_metric(self, metric: object, /) -> float | None:
        """
        Extract a single-list metric from the per-list measurement result (if
        applicable).

        Returns:
            The per-list metric, or ``None`` if this metric does not compute
            per-list metrics.
        """
        return None

    @abstractmethod
    def global_aggregate(self, values: list[object], /) -> float:
        """
        Aggregate list metrics to compute a global value.
        """
        raise NotImplementedError()
