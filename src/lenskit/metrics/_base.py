# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol

import numpy as np
import pyarrow as pa

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

    @abstractmethod
    def measure_list(self, output: ItemList, test: ItemList, /) -> object:
        """
        Compute measurements for a single list.
        """
        raise NotImplementedError()

    def extract_list_metrics(self, metric: object, /) -> float | dict[str, float] | None:
        """
        Extract per-list metric(s) from the per-list measurement result.

        Returns:
            A float, a dictionary of metric names and values, or None.
        """
        return None

    @abstractmethod
    def summarize(
        self, values: list[object] | np.ndarray | pa.Array | pa.ChunkedArray, /
    ) -> float | dict[str, float]:
        """
        Aggregate list metrics to compute one or more global summary values.

        Returns:
            A single numeric summary (float), or a dictionary of named summary values.
        """
        raise NotImplementedError()


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

    def summarize(self, values: list[object] | np.ndarray | pa.Array | pa.ChunkedArray, /) -> float:
        if isinstance(values, (pa.Array, pa.ChunkedArray)):
            values = values.to_numpy()
        elif not isinstance(values, np.ndarray):
            values = np.array(values)
        return float(np.nanmean(values))


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
    Deprecated base class for decomposed metrics.
    .. deprecated:: 2025.4
        This class is deprecated and its functionality has been moved to :class:`Metric`.
        It is scheduled for removal in 2026.

    Base class for metrics that measure entire runs through flexible
    aggregations of per-list intermediate measurements.  They can optionally
    extract individual-list metrics from the per-list measurements.

    Stability:
        Full
    """

    def measure_list(self, output: ItemList, test: ItemList, /) -> object:
        return self.compute_list_data(output, test)

    def extract_list_metrics(self, metric: object, /) -> float | dict[str, float] | None:
        return self.extract_list_metric(metric)

    def summarize(
        self, values: list[object] | np.ndarray | pa.Array | pa.ChunkedArray, /
    ) -> float | dict[str, float]:
        if isinstance(values, (pa.Array, pa.ChunkedArray)):
            values = values.to_numpy()
        elif not isinstance(values, np.ndarray):
            values = np.array(values)

        return self.global_aggregate(list(values))

    def compute_list_data(self, output: ItemList, test: ItemList, /) -> object:
        """
        Compute measurements for a single list.

        Use `measure_list` in `Metric` for new implementations.
        """
        raise NotImplementedError()

    def extract_list_metric(self, metric: object, /) -> float | None:
        """
        Extract a single-list metric from the per-list measurement result (if
        applicable).

        Returns:
            The per-list metric, or ``None`` if this metric does not compute
            per-list metrics.

        Use `extract_list_metrics` in `Metric` for new implementations.
        """
        return None

    def global_aggregate(self, values: list[object], /) -> float:
        """
        Aggregate list metrics to compute a global value.

        Use `summarize` in `Metric` for new implementations.
        """
        raise NotImplementedError()
