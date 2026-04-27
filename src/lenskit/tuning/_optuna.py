# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from os import fspath

import optuna
from optuna import Study, Trial
from optuna.pruners import MedianPruner
from optuna.storages.journal import JournalFileBackend, JournalStorage
from optuna.trial import TrialState
from pydantic_core import to_json
from structlog.stdlib import BoundLogger

from lenskit.logging import Task, get_logger, item_progress
from lenskit.pipeline import Pipeline, PipelineBuilder, PipelineConfig
from lenskit.pipeline.components import Component, Placeholder
from lenskit.pipeline.nodes import ComponentConstructorNode, ComponentInstanceNode
from lenskit.random import make_seed
from lenskit.training import ModelTrainer, TrainingOptions, UsesTrainer

from ._base import BasePipelineTuner
from ._measure import measure_pipeline
from .spec import SearchSpace

_log = get_logger(__name__)


class PipelineTuner(BasePipelineTuner):
    """
    Default pipeline tuner using Optuna.
    """

    log: BoundLogger

    @property
    def pipeline(self) -> PipelineConfig:
        assert isinstance(self.spec.pipeline, PipelineConfig)
        return self.spec.pipeline

    def run(self):
        study = optuna.create_study(
            storage=JournalStorage(JournalFileBackend(fspath(self.out_dir / "optuna.log"))),
            pruner=MedianPruner(),
        )
        # run and parallelize
        pass

    def _run_trial(self, study: Study):
        """
        Run a single trial of the hyperparameter tuning.
        """
        trial = study.ask()
        config = self._ask_config(trial)
        name = self.pipeline.meta.name
        self.log = _log.bind(pipeline=name, dataset=self.data.name)

        try:
            if self.iterative:
                self._run_iter_trial(study, trial, config)
            else:
                self._run_simple_trial(study, trial, config)
        except Exception as e:
            self.log.info("simple trial failed", trial_num=trial.number, exc_info=e)
            study.tell(trial, state=TrialState.FAIL)

    def _run_iter_trial(self, study: Study, trial: Trial, config):
        self.log.info("starting iterative trial", config=config)
        comp_name = self.spec.component_name
        assert comp_name is not None

        with Task(
            label=f"tune {self.pipeline.meta.name}",
            tags=["tune"],
            reset_hwm=True,
            subprocess=True,
        ):
            pipeline = self._pretrain_iter_pipeline(comp_name, config)
            comp_node = pipeline.node(comp_name)

            assert isinstance(comp_node, ComponentInstanceNode)
            scorer = comp_node.component

            assert isinstance(scorer, Component)
            assert isinstance(scorer, UsesTrainer)

            self.log.info("creating model trainer", config=scorer.config)
            options = TrainingOptions(
                rng=make_seed(self.random_seed, to_json(scorer.dump_config()))
            )
            trainer = scorer.create_trainer(self.data.train, options)

            with item_progress(
                f"Trial {trial.number} epochs",
                total=self.spec.search.max_epochs,
                fields={self.metric: ":.3f"},
            ) as pb:
                for i in range(self.spec.search.max_epochs):
                    metrics = self._trial_epoch(i, pipeline, trainer)
                    mv = metrics[self.metric]
                    pb.update(**{self.metric: mv})
                    trial.report(mv, i)
                    if trial.should_prune():
                        self.log.info(f"pruning after trial {trial.number}")
                        study.tell(trial, state=TrialState.PRUNED)
                        return

                study.tell(trial, mv)

    def _pretrain_iter_pipeline(self, comp_name: str, config):
        self.log.debug("setting up base pipeline")
        pb = PipelineBuilder.from_config(self.pipeline)
        comp_node = pb.node(comp_name)
        assert isinstance(comp_node, ComponentConstructorNode)
        assert isinstance(comp_node.constructor, type)

        pb.replace_component(comp_name, Placeholder)
        wrapper_pipe = pb.build()

        self.log.info("pre-training pipeline")
        wrapper_pipe.train(self.data.train)

        self.log.debug("adding untrained scorer", config=config)
        cfg_pipe = self.pipeline.merge_component_configs({comp_name: config})
        pb = PipelineBuilder.from_pipeline(wrapper_pipe)
        pb.replace_component(
            comp_name, comp_node.constructor, cfg_pipe.components[comp_name].config
        )
        return pb.build()

    def _trial_epoch(self, epoch: int, pipeline: Pipeline, trainer: ModelTrainer):
        elog = self.log.bind(epoch=epoch)
        with Task(f"epoch {epoch}", tags=["tuning", "epoch"]) as e_task:
            elog.debug("beginning training epoch")
            with Task(f"training epoch {epoch}", tags=["tuning", "epoch", "train"]) as t_task:
                vals = trainer.train_epoch()
            elog.debug("epoch training finished", result=vals, duration=t_task.friendly_duration)

            elog.debug("generating recommendations", n_queries=len(self.data.test))
            with Task(f"measuring epoch {epoch}", tags=["tuning", "epoch", "measure"]) as m_task:
                metrics = measure_pipeline(self.spec, pipeline, self.data.test)

            metrics = {"epoch": epoch} | metrics
            metrics["epoch_train_s"] = t_task.duration
            metrics["epoch_measure_s"] = m_task.duration
            elog.debug("epoch measurement finished", duration=m_task.friendly_duration)

        elog.info("epoch complete", duration=e_task.friendly_duration)
        return metrics

    def _run_simple_trial(self, study: Study, trial: Trial, config):
        log = _log.bind(pipeline=self.pipeline.meta.name, trial_num=trial.number)
        log.info("running simple trial", config=config)
        comp_name = self.spec.component_name
        assert comp_name is not None

        cfg = self.pipeline.merge_component_configs({comp_name: config})
        pipe = Pipeline.from_config(cfg)
        rng = make_seed(self.random_seed, to_json(config))

        with Task(
            label=f"train {pipe.name}",
            tags=["tune", "train"],
            reset_hwm=True,
            subprocess=True,
        ) as train_task:
            pipe.train(self.data.train, TrainingOptions(rng=rng))

        with Task(
            label=f"measure {pipe.name}",
            tags=["tune", "recommend"],
            reset_hwm=True,
            subprocess=True,
        ) as test_task:
            results = measure_pipeline(self.spec, pipe, self.data.test, train_task, test_task)

        study.tell(trial, results[self.metric], state=TrialState.COMPLETE)

    def _ask_config(self, trial: Trial):
        # we have exactly one
        for space in self.spec.space.values():
            return _ask_space(trial, space)


def _ask_space(trial: Trial, space: SearchSpace, *, prefix: str = ""):
    out = {}
    for name, spec in space.items():
        if isinstance(spec, dict):
            out[name] = _ask_space(trial, spec, prefix=f"{prefix}{name}.")
        elif spec.type == "int" and spec.scale == "uniform":
            assert isinstance(spec.min, int)
            assert isinstance(spec.max, int)
            out[name] = trial.suggest_int(prefix + name, spec.min, spec.max)
        elif spec.type == "int" and spec.scale == "log":
            assert isinstance(spec.min, int)
            assert isinstance(spec.max, int)
            out[name] = trial.suggest_int(prefix + name, spec.min, spec.max, log=True)
        elif spec.type == "float" and spec.scale == "uniform":
            out[name] = trial.suggest_float(prefix + name, spec.min, spec.max)
        elif spec.type == "float" and spec.scale == "log":
            out[name] = trial.suggest_float(prefix + name, spec.min, spec.max, log=True)

    return out
