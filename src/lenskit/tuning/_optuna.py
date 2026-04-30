# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from os import fspath

import optuna
from optuna import Study, Trial
from optuna.pruners import BasePruner, MedianPruner
from optuna.storages.journal import JournalFileBackend, JournalStorage
from optuna.study import StudyDirection
from optuna.trial import TrialState
from pydantic_core import to_json
from structlog.stdlib import BoundLogger

from lenskit.logging import Task, get_logger, item_progress
from lenskit.pipeline import Pipeline, PipelineBuilder
from lenskit.pipeline.components import Component, Placeholder
from lenskit.pipeline.nodes import ComponentConstructorNode, ComponentInstanceNode
from lenskit.random import make_seed
from lenskit.training import ModelTrainer, TrainingOptions, UsesTrainer

from ._base import BasePipelineTuner, TuneResults
from ._measure import measure_pipeline
from ._stopping import PlateauStopRule
from .spec import SearchSpace, TuningSpec

_log = get_logger(__name__)


class PipelineTuner(BasePipelineTuner):
    """
    Default pipeline tuner using Optuna.
    """

    log: BoundLogger

    def run(self) -> OptunaTuneResults:
        self.out_dir.mkdir(exist_ok=True, parents=True)
        study = optuna.create_study(
            storage=JournalStorage(JournalFileBackend(fspath(self.out_dir / "optuna.log"))),
            pruner=MedianPruner(),
            direction=StudyDirection.MINIMIZE if self.mode == "min" else StudyDirection.MAXIMIZE,
        )
        # TODO: add parallelism support

        self.log = _log.bind(pipeline=self.pipeline.meta.name, dataset=self.data.name)
        self.log.info("beginning hyperparameter search", max_points=self.spec.search.max_points)
        with Task(f"tune {self.pipeline.meta.name}", tags=["tune"], reset_hwm=True) as task:
            task.save_to_file(self.out_dir / "task.json")
            assert self.spec.search.max_points is not None
            with item_progress("Search trials", total=self.spec.search.max_points) as pb:
                for trial_no in range(self.spec.search.max_points):
                    trial = study.ask()
                    self._run_trial(study, trial)
                    pb.update()

        self.log.info("finished tuning in %s", task.friendly_duration)
        df = study.trials_dataframe()
        df.to_csv(self.out_dir / "trials.csv")

        return OptunaTuneResults(self.spec, df, task=task)

    def _run_trial(self, study: Study, trial: Trial):
        """
        Run a single trial of the hyperparameter tuning.
        """

        config = self._ask_config(trial)
        self.log = self.log.bind(trial_num=trial.number)

        with Task(
            label=f"trial {trial.number} to tune {self.pipeline.meta.name}",
            tags=["tune", "trial"],
            reset_hwm=True,
        ):
            try:
                if self.iterative:
                    self._run_iter_trial(study, trial, config)
                else:
                    self._run_simple_trial(study, trial, config)
            except Exception as e:
                self.log.info("simple trial failed", exc_info=e)
                study.tell(trial, state=TrialState.FAIL)

    def _run_iter_trial(self, study: Study, trial: Trial, config):
        self.log.info("starting iterative trial", config=config)
        comp_name = self.spec.component_name
        assert comp_name is not None

        with Task(
            label=f"setup trial {trial.number} to tune {self.pipeline.meta.name}",
            tags=["tune", "setup"],
            # reset_hwm=True,
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
            fields={self.metric: ".3f"},
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
        with Task(f"epoch {epoch}", tags=["tune", "epoch"]) as e_task:
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

        elog.info(
            "epoch complete, %s=%.3f",
            self.metric,
            metrics[self.metric],
            duration=e_task.friendly_duration,
        )
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
            # reset_hwm=True,
        ) as train_task:
            pipe.train(self.data.train, TrainingOptions(rng=rng))

        with Task(
            label=f"measure {pipe.name}",
            tags=["tune", "recommend"],
            # reset_hwm=True,
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


@dataclass
class OptunaTuneResults(TuneResults):
    spec: TuningSpec
    study: Study

    def num_trials(self):
        return len(self.study.trials)

    def best_config(self):
        raise NotImplementedError()

    def best_result(self):
        raise NotImplementedError()


class CompositePruner(BasePruner):
    """
    Custom pruner that prunes when either the median rule or the plateau stopper
    stops.
    """

    median: BasePruner

    def __init__(self):
        self.median = MedianPruner(n_min_trials=3, n_warmup_steps=3)

    def prune(self, study, trial):
        step = trial.last_step
        log = _log.bind(trial_num=trial.number, epoch=step)
        log.debug("checking whether to prune trial")
        if self.median.prune(study, trial):
            log.debug("pruning by median rule")
            return True

        if step is None:
            return False

        plateau = PlateauStopRule(
            mode="min" if study.direction == StudyDirection.MINIMIZE else "max"
        )
        metrics = [
            trial.intermediate_values[i] for i in range(step + 1) if i in trial.intermediate_values
        ]
        if plateau.should_stop(metrics):
            log.debug("pruning by plateau")
            return True

        return False
