# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Simple (non-iterative) evaluation of points.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import ray
import ray.tune
import ray.tune.result
import torch
from pydantic_core import to_json
from structlog.stdlib import BoundLogger

from lenskit.batch import BatchPipelineRunner
from lenskit.logging import Task, get_logger
from lenskit.logging.worker import send_task
from lenskit.pipeline import Component, Pipeline, PipelineBuilder
from lenskit.pipeline.components import Placeholder
from lenskit.pipeline.nodes import ComponentConstructorNode, ComponentInstanceNode
from lenskit.random import make_seed
from lenskit.splitting.split import TTSplit
from lenskit.state import ParameterContainer
from lenskit.training import ModelTrainer, TrainingOptions, UsesTrainer

from .job import TuningJobData
from .metrics import measure_pipeline

_log = get_logger(__name__)


class IterativeEval(ray.tune.Trainable):
    """
    A simple hyperparameter point evaluator using non-iterative model training.
    """

    job: TuningJobData
    log: BoundLogger
    data: TTSplit
    task: Task

    pipeline: Pipeline
    trainer: ModelTrainer

    def setup(self, config, job):
        self.job = job
        self.data = self.job.data
        name = self.job.pipeline.meta.name

        self.log = _log.bind(pipeline=name, dataset=self.job.data_name)

        self.task = Task(
            label=f"tune {name}",
            tags=["tune"],
            reset_hwm=True,
            subprocess=True,
        )

        comp_name = self.job.spec.component_name
        assert comp_name is not None
        self.log.info("setting up base pipeline")
        pb = PipelineBuilder.from_config(self.job.pipeline)
        comp_node = pb.node(comp_name)
        assert isinstance(comp_node, ComponentConstructorNode)
        assert isinstance(comp_node.constructor, type)

        pb.replace_component(comp_name, Placeholder)
        self.pipeline = pb.build()

        self.log.info("pre-training pipeline")
        self.pipeline.train(self.data.train)

        self.log.info("adding scorer", config=config)
        pb = PipelineBuilder.from_pipeline(self.pipeline)
        comp_cfg = self.job.pipeline.components[comp_name].config or {}
        pb.replace_component(comp_name, comp_node.constructor, comp_cfg | config)
        self.pipeline = pb.build()
        comp_node = self.pipeline.node(comp_name)

        assert isinstance(comp_node, ComponentInstanceNode)
        self.scorer = comp_node.component

        assert isinstance(self.scorer, Component)
        assert isinstance(self.scorer, UsesTrainer)

        send_task(self.task)

        self.log.info("creating model trainer", config=self.scorer.config)
        options = TrainingOptions(
            rng=make_seed(self.job.random_seed, to_json(self.scorer.dump_config()))
        )
        self.trainer = self.scorer.create_trainer(self.data.train, options)
        send_task(self.task)

        self.runner = BatchPipelineRunner(n_jobs=1)  # single-threaded inside tuning
        self.runner.recommend()
        if self.pipeline.node("rating-predictor", missing=None) is not None:
            self.runner.predict()

    def step(self):
        epoch = self.iteration
        epoch_limit = self.job.spec.search.max_epochs
        if epoch > epoch_limit:
            return {ray.tune.result.DONE: True}

        elog = self.log.bind(epoch=epoch)
        with Task(f"epoch {self.iteration}", tags=["tuning", "epoch"]) as e_task:
            elog.debug("beginning training iteration")
            with Task(
                f"training epoch {self.iteration}", tags=["tuning", "epoch", "train"]
            ) as t_task:
                vals = self.trainer.train_epoch()
            elog.debug("epoch training finished", result=vals, duration=t_task.friendly_duration)

            elog.debug("generating recommendations", n_queries=len(self.data.test))
            with Task(
                f"measuring epoch {self.iteration}", tags=["tuning", "epoch", "measure"]
            ) as m_task:
                metrics = measure_pipeline(self.job.spec, self.pipeline, self.data.test)
                metrics["max_epochs"] = epoch_limit
                if epoch == epoch_limit:
                    metrics[ray.tune.result.DONE] = True

            metrics["epoch_train_s"] = t_task.duration
            metrics["epoch_measure_s"] = m_task.duration
            elog.debug("epoch measurement finished", duration=m_task.friendly_duration)

        send_task(self.task)
        elog.info("epoch complete", duration=e_task.friendly_duration)
        return metrics

    def cleanup(self):
        self.trainer.finalize()

    def save_checkpoint(self, checkpoint_dir: str):
        cpdir = Path(checkpoint_dir)
        log = self.log.bind(epochs=self.iteration)
        if isinstance(self.trainer, ParameterContainer):
            log.info("saving checkpoint")
            torch.save(
                self.trainer.get_parameters(),
                cpdir / "model.pt",
                pickle_protocol=pickle.HIGHEST_PROTOCOL,
            )
        else:
            log.warning("trainer does not implement ParameterContainer, pickling")
            with open(cpdir / "model.pkl", "wb") as pkf:
                pickle.dump(self.trainer, pkf)

    def load_checkpoint(self, checkpoint_dir: str):
        cpdir = Path(checkpoint_dir)

        self.log.info("resuming from checkpoint", epochs=self.iteration)

        ptf = cpdir / "model.pt"
        if ptf.exists():
            assert isinstance(self.trainer, ParameterContainer)
            self.trainer.load_parameters(torch.load(ptf, weights_only=False))
        else:
            with open(cpdir / "model.pkl", "rb") as pkf:
                self.trainer = pickle.load(pkf)
