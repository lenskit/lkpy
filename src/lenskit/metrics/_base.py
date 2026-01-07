# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol

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

    This class defines the interface for metrics. Subclasses should implement
    the `measure_list` method to compute metric values.

    The `summarize()` method has a default implementation that computes the
    mean of the per-list metric values, but subclasses can override it to provide
    more appropriate summary statistics.

    Stability:
        Full

    .. note::

        For simplicity in the analysis code, you cannot simply implement the
        properties of this class on an arbitrary class in order to implement a
        metric with all available behavior such as labeling and defaults;
        you must actually extend this class. This requirement may be relaxed
        in the future.

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
    def measure_list(self, output: ItemList, test: ItemList, /) -> Any:
        """
        Compute measurements for a single list.

        Returns:
            - A float for simple metrics
            - Intermediate data for decomposed metrics
            - A dict mapping metric names to values for multi-metric classes
        """
        raise NotImplementedError()  # pragma: no cover

    def extract_list_metrics(self, data: Any, /) -> float | dict[str, float] | None:
        """
        Extract per-list metric(s) from intermediate measurement data.

        Returns:
            - A float for simple metrics
            - A dict mapping metric names to values for multi-metric classes
            - None if no per-list metrics are available
        """
        return None

    @abstractmethod
    def summarize(self, values: list[Any] | pa.Array | pa.ChunkedArray, /) -> dict[str, float]:
        """
        Aggregate intermediate values into summary statistics.

        Returns:
            A dictionary of summary statistics.
        """
        raise NotImplementedError()  # pragma: no cover


class ListMetric(Metric):
    """
    Base class for metrics that measure individual recommendation (or
    prediction) lists, and whose results may be aggregated to compute overall
    metrics.

    For prediction metrics, this is *macro-averaging*.

    Default behavior:
        Implements `summarize()` by averaging per-list results (mean, ignoring NaNs).

    This class implements the Metric interface in terms of the measure_list method.

    Stability:
        Full
    """

    default: ClassVar[float | None] = 0.0

    @abstractmethod
    def measure_list(self, output: ItemList, test: ItemList, /) -> float:
        """
        Compute the metric value for a single result list.

        Individual metric classes need to implement this method.
        """
        raise NotImplementedError()  # pragma: no cover

    def extract_list_metrics(self, data: Any, /) -> float:
        """
        Return the given per-list metric result.
        """
        return data

    def summarize(
        self, values: list[Any] | pa.Array | pa.ChunkedArray, /
    ) -> dict[str, float | None]:
        """
        Summarize per-list metric values

        Returns:
            A dictionary containing mean, median, and std.
        """
        if isinstance(values, (pa.Array, pa.ChunkedArray)):
            values = values.to_pylist()

        numeric_values = [
            float(v) for v in values if isinstance(v, (int, float, np.integer, np.floating))
        ]

        if not numeric_values:
            return {"mean": None, "median": None, "std": None}

        arr = np.array(numeric_values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }


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
        raise NotImplementedError()  # pragma: no cover

    def measure_list(self, output: ItemList, test: ItemList, /) -> Any:
        raise NotImplementedError("Global metrics don't support per-list measurement")

    def summarize(self, values: list[Any] | pa.Array | pa.ChunkedArray, /) -> float:
        raise NotImplementedError("Global metrics should implement measure_run instead")


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

    def measure_list(self, output: ItemList, test: ItemList, /) -> Any:
        return self.compute_list_data(output, test)

    def extract_list_metrics(self, data: Any, /) -> float | None:
        return self.extract_list_metric(data)

    def summarize(self, values: list[Any] | pa.Array | pa.ChunkedArray, /) -> dict[str, float]:
        if isinstance(values, (pa.Array, pa.ChunkedArray)):
            values = values.to_pylist()
        result = self.global_aggregate(values)
        if isinstance(result, (float, int, np.floating, np.integer)):
            return {"value": float(result)}
        return result

    @abstractmethod
    def compute_list_data(self, output: ItemList, test: ItemList, /) -> Any:
        """
        Compute measurements for a single list.

        Use `measure_list` in `Metric` for new implementations.
        """
        raise NotImplementedError()  # pragma: no cover

    def extract_list_metric(self, data: Any, /) -> float | None:
        """
        Extract a single-list metric from the per-list measurement result (if
        applicable).

        Returns:
            The per-list metric, or ``None`` if this metric does not compute
            per-list metrics.

        Implement :meth:`Metric.extract_list_metrics` in new implementations.
        """
        return None

    @abstractmethod
    def global_aggregate(self, values: list[Any], /) -> float | dict[str, float]:
        """
        Aggregate list metrics to compute a global value.

        Implement :meth:`Metric.summarize` in new implementations.
        """
        raise NotImplementedError()  # pragma: no cover
