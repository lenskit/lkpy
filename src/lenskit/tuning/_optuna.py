# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from os import fspath

import numpy as np
import optuna
import pandas as pd
from optuna import Study, Trial
from optuna.pruners import BasePruner, MedianPruner
from optuna.storages.journal import JournalFileBackend, JournalStorage
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState
from pydantic import JsonValue
from pydantic_core import to_json
from structlog.stdlib import BoundLogger
from typing_extensions import override

from lenskit.data import unflatten_dict
from lenskit.logging import Task, get_logger, item_progress, trace
from lenskit.logging.tasks import add_context_task
from lenskit.parallel import NestedPool
from lenskit.pipeline import Pipeline, PipelineBuilder
from lenskit.pipeline.components import Component, Placeholder
from lenskit.pipeline.nodes import ComponentConstructorNode, ComponentInstanceNode
from lenskit.random import int_seed, make_seed
from lenskit.schemas.tuning import SearchParam, SearchSpace, TuningSpec
from lenskit.training import ModelTrainer, TrainingOptions, UsesTrainer

from ._base import BasePipelineTuner, TuneResults
from ._measure import measure_pipeline
from ._stopping import PlateauStopRule

_log = get_logger(__name__)


class PipelineTuner(BasePipelineTuner):
    """
    Default pipeline tuner using Optuna.
    """

    log: BoundLogger

    def run(self) -> OptunaTuneResults:
        self.out_dir.mkdir(exist_ok=True, parents=True)
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=int_seed(self.random_seed)),
            storage=JournalStorage(JournalFileBackend(fspath(self.out_dir / "optuna.log"))),
            pruner=CompositePruner(),
            direction=StudyDirection.MINIMIZE if self.mode == "min" else StudyDirection.MAXIMIZE,
        )

        self.log = _log.bind(pipeline=self.pipeline.meta.name, dataset=self.data.name)
        self.log.info(
            "beginning hyperparameter search", points=self.spec.search.num_search_points()
        )
        with Task(f"tune {self.pipeline.meta.name}", tags=["tune"], reset_hwm=True) as task:
            task.save_to_file(self.out_dir / "task.json")
            self._run_study(study)

        self.log.info("finished tuning in %s", task.friendly_duration)
        df = study.trials_dataframe()
        df.to_csv(self.out_dir / "trials.csv", index=False)

        if self.iterative:
            idf = pd.DataFrame.from_records(
                {"trial": t.number, "epoch": e, self.metric: val}
                for t in study.trials
                for (e, val) in t.intermediate_values.items()
            )
            idf.to_csv(self.out_dir / "trial-epochs.csv", index=False)

        return OptunaTuneResults(spec=self.spec, study=study, iterative=self.iterative, task=task)

    def _run_study(self, study: Study):
        self._enqueue_defaults(study)
        npts = self.spec.search.num_search_points()
        task = Task.current()
        with item_progress("Search trials", total=npts) as pb:
            if self.settings.jobs and self.settings.jobs > 1:
                with ThreadPoolExecutor(
                    self.settings.jobs, "lk-tune", initializer=add_context_task, initargs=(task,)
                ) as pool:
                    tasks = [
                        pool.submit(lambda: self._run_trial(study, nested_pool=True))
                        for _i in range(npts)
                    ]
                    while tasks:
                        done, tasks = wait(tasks, return_when=FIRST_COMPLETED)
                        for t in done:
                            if exc := t.exception():
                                _log.error("tuning trial failed", exc_info=exc)
                                if self.spec.search.error_action == "abort":
                                    pool.shutdown(False, cancel_futures=True)
                                    raise exc

                            pb.update()
            else:
                for trial_no in range(npts):
                    self._run_trial(study)
                    pb.update()

    def _run_trial(self, study: Study, *, nested_pool: bool = False):
        """
        Run a single trial of the hyperparameter tuning.
        """
        if nested_pool:
            with NestedPool():
                return self._run_trial(study)

        trial = study.ask()
        self.log = self.log.bind(trial_num=trial.number)
        config = self._ask_config(trial)

        with Task(
            label=f"trial {trial.number} to tune {self.pipeline.meta.name}",
            tags=["tune", "trial"],
            reset_hwm=True,
        ):
            self.log.debug("beginning search", config=config)
            try:
                if self.iterative:
                    self._run_iter_trial(study, trial, config)
                else:
                    self._run_simple_trial(study, trial, config)
            except Exception as e:
                self.log.info("trial failed", exc_info=e)
                study.tell(trial, state=TrialState.FAIL)
                raise e

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
                    self.log.info(f"pruning after epoch {i}")
                    # we need to report as complete, not pruned, so they are
                    # considered as valid results.
                    study.tell(trial, mv, state=TrialState.COMPLETE)
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

    def _enqueue_defaults(self, study: Study):
        """
        Enqueue the component's default hyperparameters into the study.
        """
        self.log.debug(
            "instantiating pipeline to extract defaults", component=self.spec.component_name
        )
        pipe = Pipeline.from_config(self.pipeline)
        assert self.spec.component_name is not None
        comp = pipe.component(self.spec.component_name)
        assert comp is not None
        if not isinstance(comp, Component):
            self.log.warn("component is not pipeline", component=self.spec.component_name)
            return

        config = _extract_defaults(self.spec.space[self.spec.component_name], comp.dump_config())
        self.log.info("enqueueing default point", config=config)
        study.enqueue_trial(
            config,
            user_attrs={
                "memo": "initial default configuration",
                "component": self.spec.component_name,
            },
        )

    def _ask_config(self, trial: Trial):
        # we have exactly one
        for space in self.spec.space.values():
            cfg = _ask_space(trial, space)
            self.log.info("sampled search point", config=cfg)
            return cfg


def _ask_space(trial: Trial, space: SearchSpace, *, prefix: str = ""):
    out = {}
    for name, spec in space.items():
        if isinstance(spec, dict):
            out[name] = _ask_space(trial, spec, prefix=f"{prefix}{name}.")
        elif spec.type == "int" and spec.scale == "uniform":
            assert isinstance(spec.min, int)
            assert isinstance(spec.max, int)
            out[name] = trial.suggest_int(prefix + name, spec.min, spec.max)
            trace(
                _log, "sampled int", name=prefix + name, min=spec.min, max=spec.max, value=out[name]
            )
        elif spec.type == "int" and spec.scale == "log":
            assert isinstance(spec.min, int)
            assert isinstance(spec.max, int)
            out[name] = trial.suggest_int(prefix + name, spec.min, spec.max, log=True)
            trace(
                _log,
                "sampled int/log2",
                name=prefix + name,
                min=spec.min,
                max=spec.max,
                value=out[name],
            )
        elif spec.type == "int" and spec.scale == "pow2":
            min = int(math.log2(spec.min))
            max = int(math.log2(spec.max))
            p = trial.suggest_int(prefix + name, min, max)
            out[name] = 2**p
            trace(
                _log,
                "sampled int/pow2",
                name=prefix + name,
                min=min,
                max=max,
                power=p,
                value=out[name],
            )
        elif spec.type == "float" and spec.scale == "uniform":
            out[name] = trial.suggest_float(prefix + name, spec.min, spec.max)
            trace(
                _log,
                "sampled float",
                name=prefix + name,
                min=spec.min,
                max=spec.max,
                value=out[name],
            )
        elif spec.type == "float" and spec.scale == "log":
            out[name] = trial.suggest_float(prefix + name, spec.min, spec.max, log=True)
            trace(
                _log,
                "sampled float/log",
                name=prefix + name,
                min=spec.min,
                max=spec.max,
                value=out[name],
            )
        elif spec.type == "bool":
            out[name] = trial.suggest_categorical(prefix + name, [False, True])
            trace(_log, "sampled bool", name=prefix + name, value=out[name])
        elif spec.type == "choice":
            out[name] = trial.suggest_categorical(prefix + name, spec.choices)
            trace(_log, "sampled choice", name=prefix + name, value=out[name])
        else:  # pragma: nocover
            raise ValueError(f"unsupported configuration {space}")

    return out


def _extract_defaults(
    space: dict[str, SearchSpace],
    config: dict[str, JsonValue],
    *,
    out: dict[str, JsonValue] | None = None,
    prefix: str = "",
) -> dict[str, JsonValue]:
    # FIXME this whole function is ugly but seems to work
    if out is None:
        out = {}
    for k, v in space.items():
        if isinstance(v, dict):
            _extract_defaults(v, config[k], out=out, prefix=f"{prefix}{k}.")  # type: ignore
        elif isinstance(config, dict):
            assert isinstance(v, SearchParam)
            try:
                out[k] = config[k]
            except KeyError as e:
                if k.endswith("_exp"):
                    out[k] = int(math.log2(config[k[:-4]]))  # type: ignore
                else:
                    raise e
            if v.scale == "pow2":
                out[k] = int(math.log2(out[k]))  # type: ignore
        else:
            # FIXME this is an ugly hack for single / multiple configs
            out[k] = config
    return out


@dataclass
class OptunaTuneResults(TuneResults):
    spec: TuningSpec
    study: Study
    iterative: bool

    def num_trials(self):
        return len(self.study.trials)

    @override
    def trials(self) -> Iterable[dict[str, JsonValue]]:
        for trial in self.study.trials:
            if trial.state != TrialState.FAIL:
                yield self._trial_result(trial)

    @override
    def epochs(self) -> Iterable[dict[str, JsonValue]]:
        if not self.iterative:
            return []

        for n, trial in enumerate(self.study.trials):
            if trial.state == TrialState.FAIL:
                continue

            for i in range(trial.last_step + 1):
                out_row = {
                    "trial": n,
                    "epoch": i,
                    self.spec.search.metric: trial.intermediate_values.get(i),
                }
                yield out_row

    def best_config(self):
        best = self.study.best_trial
        cfg = unflatten_dict(best.params)
        if self.iterative:
            vals = [best.intermediate_values.get(i) for i in range(best.last_step + 1)]
            match self.study.direction:
                case StudyDirection.MAXIMIZE:
                    cfg["epochs"] = np.argmax(vals).item() + 1
                case StudyDirection.MINIMIZE:
                    cfg["epochs"] = np.argmin(vals).item() + 1
                case _:  # pragma: nocover
                    raise RuntimeError("unexpected study direction")

        return cfg

    def best_result(self) -> dict[str, JsonValue]:
        return self._trial_result(self.study.best_trial)

    def _trial_result(
        self, trial: FrozenTrial, *, include_config: bool = True
    ) -> dict[str, JsonValue]:
        assert self.spec.search.metric is not None
        result: dict[str, JsonValue]

        if self.iterative:
            assert trial.last_step is not None
            vals = [trial.intermediate_values.get(i) for i in range(trial.last_step + 1)]
            epoch = np.argmax(vals).item()
            result = {
                self.spec.search.metric: vals[epoch],
                "epochs": epoch,
                "attempted_epochs": len(trial.intermediate_values),
                "final_metric": trial.value,
            }
        else:
            result = {self.spec.search.metric: trial.value}

        if include_config:
            result["config"] = self._trial_config(trial)
        return result

    def _trial_config(self, trial: FrozenTrial) -> dict[str, JsonValue]:
        cfg = unflatten_dict(trial.params)
        if self.iterative:
            assert trial.last_step is not None
            vals = [trial.intermediate_values.get(i) for i in range(trial.last_step + 1)]
            cfg["epochs"] = np.argmax(vals).item()

        return cfg


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
