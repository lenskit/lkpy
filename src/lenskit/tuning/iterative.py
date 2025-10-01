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
from codex.models import load_model
from codex.random import extend_seed
from codex.recpipe import base_pipeline, replace_scorer
from codex.runlog import CodexTask, ScorerModel
from pydantic_core import to_json
from structlog.stdlib import BoundLogger

from lenskit.batch import BatchPipelineRunner
from lenskit.logging import Task, get_logger
from lenskit.logging.worker import send_task
from lenskit.pipeline import Component
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
    task: CodexTask
    trainer: ModelTrainer

    def setup(self, config, job):
        self.job = job
        self.mod_def = load_model(self.job.model_name)
        factory = self.job.factory
        self.data = self.job.data
        self.log = _log.bind(model=self.job.model_name, dataset=self.job.data_name)

        self.task = CodexTask(
            label=f"tune {self.mod_def.name}",
            tags=["tuning"],
            reset_hwm=True,
            subprocess=True,
            scorer=ScorerModel(name=self.mod_def.name, config=config),
            data=self.job.data_info,
        )

        self.log.info("configuring scorer", config=config)
        self.scorer = self.mod_def.instantiate(config, factory)
        assert isinstance(self.scorer, Component)
        assert isinstance(self.scorer, UsesTrainer)
        pipe = base_pipeline(self.job.model_name, predicts_ratings=self.mod_def.is_predictor)

        self.log.info("pre-training pipeline")
        pipe.train(self.data.train)
        self.pipe = replace_scorer(pipe, self.scorer)
        send_task(self.task)

        self.log.info("creating model trainer", config=self.scorer.config)
        options = TrainingOptions(
            rng=extend_seed(self.job.random_seed, to_json(self.scorer.dump_config()))
        )
        self.trainer = self.scorer.create_trainer(self.data.train, options)
        send_task(self.task)

        self.runner = BatchPipelineRunner(n_jobs=1)  # single-threaded inside tuning
        self.runner.recommend()
        if self.mod_def.is_predictor:
            self.runner.predict()

    def step(self):
        epoch = self.iteration
        if epoch > self.job.epoch_limit:
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
                metrics = measure_pipeline(self.mod_def, self.pipe, self.data.test)
                metrics["max_epochs"] = self.job.epoch_limit
                if epoch == self.job.epoch_limit:
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
