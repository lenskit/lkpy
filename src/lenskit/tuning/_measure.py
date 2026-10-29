# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pandas as pd

from lenskit import Pipeline
from lenskit.batch import BatchPipelineRunner
from lenskit.data import ItemListCollection
from lenskit.logging import Task, get_logger
from lenskit.metrics import (
    DCG,
    MAE,
    NDCG,
    RBP,
    RMSE,
    Hit,
    LogRBP,
    MeasurementCollector,
    RecipRank,
    measure_list,
)
from lenskit.schemas.tuning import TuningSpec

_log = get_logger(__name__)


def measure_pipeline(
    spec: TuningSpec,
    pipe: Pipeline,
    test_users: ItemListCollection,
    train_task: Task | None = None,
    test_task: Task | None = None,
):
    collector = MeasurementCollector()
    collector.add_metric(RBP())
    collector.add_metric(LogRBP())
    collector.add_metric(DCG())
    collector.add_metric(NDCG())
    collector.add_metric(NDCG(n=10))
    collector.add_metric(RecipRank())
    collector.add_metric(Hit())
    collector.add_metric(Hit(n=10))

    prediction_metrics = []

    runner = BatchPipelineRunner()
    runner.recommend(n=spec.search.list_length)
    if pipe.node("rating-predictor", missing=None) is not None:
        runner.predict()

    for result in runner.run_iter(pipe, test_users):
        log = _log.bind(key=result.key)
        log.debug("measuring recommendation list")

        recs = result.outputs["recommendations"]
        preds = result.outputs.get("predictions", None)
        test = test_users.lookup(result.key)
        assert test is not None

        collector.add_list_measurement(recs, test)

        if preds is not None:
            log.debug("measuring rating predictions")
            prediction_metrics.append(
                {"RMSE": measure_list(RMSE, preds, test), "MAE": measure_list(MAE, preds, test)}
            )

    summary = collector.summary_metrics()
    metric_names = ("RBP", "LogRBP", "DCG", "NDCG", "NDCG@10", "RecipRank", "Hit", "Hit@10")

    agg_metrics = {}

    for name in metric_names:
        if f"{name}.mean" in summary:
            agg_metrics[name] = summary[f"{name}.mean"]
        elif f"{name}.logmean" in summary:
            agg_metrics[name] = summary[f"{name}.logmean"]
        else:
            raise ValueError(f"No supported aggregate for metric {name}")

    if prediction_metrics:
        agg_metrics.update(pd.DataFrame.from_records(prediction_metrics).mean().to_dict())

    if train_task is not None:
        agg_metrics["TrainTask"] = train_task.task_id
        agg_metrics["TrainTime"] = train_task.duration
        agg_metrics["TrainCPU"] = train_task.cpu_time
    if test_task is not None:
        test_task.update_resources()
        agg_metrics["TestTask"] = test_task.task_id
        agg_metrics["TestTime"] = test_task.duration
        agg_metrics["TestCPU"] = test_task.cpu_time
    return agg_metrics
