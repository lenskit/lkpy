# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pandas as pd

from lenskit import Pipeline
from lenskit.batch import BatchPipelineRunner
from lenskit.data import GenericKey, ItemList, ItemListCollection, key_dict
from lenskit.logging import Task, get_logger
from lenskit.metrics import (
    DCG,
    MAE,
    NDCG,
    RBP,
    RMSE,
    Hit,
    MetricResult,
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
    # TODO integrate this with metric collectors
    metrics = []

    runner = BatchPipelineRunner()
    runner.recommend(n=spec.search.list_length)
    if pipe.node("rating-predictor", missing=None) is not None:
        runner.predict()

    for result in runner.run_iter(pipe, test_users):
        recs = result.outputs["recommendations"]
        preds = result.outputs.get("predictions", None)
        test = test_users.lookup(result.key)
        assert test is not None
        metrics.append(measure_recs(spec, result.key, recs, preds, test))

    metric_df = pd.DataFrame.from_records(metrics)
    metric_df = metric_df.drop(columns=["user_id"])
    agg_metrics = metric_df.mean().to_dict()

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


def measure_recs(
    spec: TuningSpec,
    key: GenericKey,
    recs: ItemList,
    preds: ItemList | None,
    test: ItemList,
):
    log = _log.bind(key=key)
    log.debug("measuring recommendation list")

    metrics: dict[str, MetricResult] = key_dict(key)
    metrics["RBP"] = measure_list(RBP, recs, test)
    metrics["DCG"] = measure_list(DCG, recs, test)
    metrics["NDCG"] = measure_list(NDCG, recs, test)
    metrics["NDCG@10"] = measure_list(NDCG, recs, test, n=10)
    metrics["RecipRank"] = measure_list(RecipRank, recs, test)
    metrics["Hit"] = measure_list(Hit, recs, test)
    metrics["Hit@10"] = measure_list(Hit, recs, test, n=10)

    if preds is not None:
        log.debug("measuring rating predictions")
        metrics["RMSE"] = measure_list(RMSE, preds, test)
        metrics["MAE"] = measure_list(MAE, preds, test)

    return metrics
