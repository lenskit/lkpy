# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import Any, Dict, List, Optional, Union

import pandas as pd

from lenskit.data import ItemList

from ._base import Metric


class MetricAccumulator:
    """
    Handles metric registration, accumulation of metric measurements
    per recommendation list, and aggregation of summary statistics.
    """

    def __init__(self):
        self._metrics: List[Metric] = []
        self._results: List[dict[str, Any]] = []

    def add_metric(self, metric: Metric) -> None:
        """
        Register a metric to be accumulated.

        Args:
            metric: An instance of a Metric subclass.
        """
        self._metrics.append(metric)

    def measure_list(
        self,
        output: ItemList,
        test: ItemList,
        user_id: Optional[Any] = None,
    ) -> None:
        """
        Measure and accumulate metrics for a single recommendation list.

        Args:
            output: The recommendation items for a user.
            test: The ground-truth items for the same user.
            user_id: Optional identifier for the user.
        """
        record = {"user_id": user_id}
        for metric in self._metrics:
            result = metric.measure_list(output, test)
            record[metric.label] = metric.extract_list_metrics(result)
        self._results.append(record)

    def list_metrics(self) -> pd.DataFrame:
        """
        Get accumulated per-list metric results.

        Returns:
            A DataFrame where each row corresponds to a user,
            and columns correspnd to metrics.
        """
        return pd.DataFrame(self._results)

    def summary_metrics(self) -> Dict[str, float]:
        """
        Compute overall summary statistics by aggregating per-list metrics.

        Returns:
            A dictionary mapping metric labels to their aggregated summary values.
        """
        summaries = {}
        df = self.list_metrics()
        for metric in self._metrics:
            values = df[metric.label].dropna().tolist()
            summaries[metric.label] = metric.summarize(values)
        return summaries
