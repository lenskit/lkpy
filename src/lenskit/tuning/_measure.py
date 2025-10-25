# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pandas as pd
from pydantic import JsonValue

from lenskit import Pipeline, predict, recommend
from lenskit.data import GenericKey, ItemList, ItemListCollection
from lenskit.data.collection import key_dict
from lenskit.logging import Task, get_logger, item_progress
from lenskit.metrics import (
    DCG,
    NDCG,
    RBP,
    RMSE,
    Hit,
    RecipRank,
    call_metric,
)

from .spec import TuningSpec

_log = get_logger(__name__)


def measure_pipeline(
    spec: TuningSpec,
    pipe: Pipeline,
    test_users: ItemListCollection,
    train_task: Task | None = None,
    test_task: Task | None = None,
):
    # TODO: integrate this with metric collectors
    metrics = []
    with item_progress("Measuring for users", total=len(test_users)) as pb:
        for key, test in test_users.items():
            metrics.append(measure_request(spec, pipe, key, test))
            pb.update()

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


def measure_request(spec: TuningSpec, pipe: Pipeline, key: GenericKey, test: ItemList):
    recs = recommend(pipe, query=key.user_id, n=spec.search.list_length)  # type: ignore
    if pipe.node("rating-predictor", missing=None) is not None:
        preds = predict(pipe, query=key.user_id, items=test)  # type: ignore
    else:
        preds = None
    return measure_list(spec, key, recs, preds, test)


def measure_list(
    spec: TuningSpec,
    key: GenericKey,
    recs: ItemList,
    preds: ItemList | None,
    test: ItemList,
):
    log = _log.bind(key=key)
    log.debug("measuring recommendation list")

    metrics: dict[str, JsonValue] = key_dict(key)
    metrics["RBP"] = float(call_metric(RBP, recs, test))
    metrics["DCG"] = float(call_metric(DCG, recs, test))
    metrics["NDCG"] = float(call_metric(NDCG, recs, test))
    metrics["RecipRank"] = float(call_metric(RecipRank, recs, test))
    metrics["Hit10"] = float(call_metric(Hit, recs, test, n=10))

    if preds is not None:
        log.debug("measuring rating predictions")
        metrics["RMSE"] = call_metric(RMSE, preds, test)

    return metrics
