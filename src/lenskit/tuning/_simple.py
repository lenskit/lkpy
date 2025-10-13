# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Simple (non-iterative) evaluation of points.
"""

from __future__ import annotations

from pydantic_core import to_json

from lenskit.logging import Task, get_logger
from lenskit.logging.worker import send_task
from lenskit.pipeline import Pipeline
from lenskit.random import make_seed
from lenskit.training import TrainingOptions

from ._job import TuningJobData
from ._measure import measure_pipeline

_log = get_logger(__name__)


class SimplePointEval:
    """
    A simple hyperparameter point evaluator using non-iterative model training.
    """

    job: TuningJobData

    def __init__(self, job: TuningJobData):
        self.job = job

    def __call__(self, config) -> dict[str, float]:
        comp_name = self.job.spec.component_name
        assert comp_name is not None
        cfg = self.job.pipeline.merge_component_configs({comp_name: config})

        pipe = Pipeline.from_config(cfg)

        data = self.job.data

        rng = make_seed(self.job.random_seed, to_json(config))
        with Task(
            label=f"train {pipe.name}",
            tags=["tune", "train"],
            reset_hwm=True,
            subprocess=True,
        ) as train_task:
            pipe.train(data.train, TrainingOptions(rng=rng))
        send_task(train_task)

        with Task(
            label=f"measure {pipe.name}",
            tags=["tune", "recommend"],
            reset_hwm=True,
            subprocess=True,
        ) as test_task:
            results = measure_pipeline(self.job.spec, pipe, data.test, train_task, test_task)
        send_task(test_task)

        return results
