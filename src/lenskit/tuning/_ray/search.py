# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import ray
import ray.tune
import ray.tune.schedulers
import ray.tune.search
from numpy.random import default_rng
from pydantic import JsonValue
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch

from lenskit.parallel import get_parallel_config
from lenskit.parallel.ray import ensure_cluster
from lenskit.pipeline.config import PipelineConfig
from lenskit.random import int_seed

from .._base import PipelineTuner, TuneResults
from ..spec import SearchSpace, TuningSpec
from .iterative import IterativeEval
from .job import TuningJobData
from .reporting import ProgressReport, StatusCallback
from .simple import SimplePointEval
from .stopper import RelativePlateauStopper


class RayPipelineTuner(PipelineTuner):
    tuner: ray.tune.Tuner

    def setup(self):
        """
        Set up to run the trainer.  After this method completes, the
        :attr:`tuner` is ready.
        """
        ensure_cluster()
        self.setup_harness()
        self.tuner = self.create_tuner()
        self.out_dir.mkdir(exist_ok=True, parents=True)

    def run(self) -> RayTuneResults:
        """
        Run the tuning job.

        Saves the results in :attr:`results`, and also returns them.
        """
        if not hasattr(self, "tuner"):
            self.setup()

        self.log.info("starting hyperparameter search")
        results = self.tuner.fit()
        self.log.info("finished hyperparameter search")

        return RayTuneResults(self.spec, self.iterative, results)

    def search_space(self):
        """
        Get the Ray search space.
        """
        # we have exactly one
        for space in self.spec.space.values():
            return _make_space(space)

    def setup_harness(self):
        self.log.info("setting up test harness")

        self.log.info("pushing data to cluster")
        data_ref = ray.put(self.data)

        self.job = TuningJobData(
            spec=self.spec,
            random_seed=self.random_seed.spawn(1)[0],
            data_name=self.data.train.name,
            data_ref=data_ref,
        )

        if self.iterative:
            harness = ray.tune.with_parameters(IterativeEval, job=self.job)
        else:
            harness = SimplePointEval(self.job)

        paracfg = get_parallel_config()

        match self.spec.search.num_cpus:
            case "threads":
                tune_cpus = paracfg.num_threads or 1
            case "backend-threads":
                tune_cpus = paracfg.num_backend_threads or 1
            case "all-threads":
                tune_cpus = paracfg.total_threads
            case int(n) if n > 0:
                tune_cpus = n
            case _:
                raise ValueError(f"invalid CPU count {self.spec.search.num_cpus}")

        self.log.info("setting up parallel tuner", cpus=tune_cpus)

        resources: dict[str, float | int] = {"CPU": tune_cpus}
        if self.spec.search.num_gpus:
            resources["GPU"] = self.spec.search.num_gpus * self.settings.gpu_mult
        self.harness = ray.tune.with_resources(harness, resources)

    @property
    def metric(self):
        metric = self.spec.search.metric
        if metric is None:
            raise RuntimeError("no metric specified")
        else:
            return metric

    def create_tuner(self) -> ray.tune.Tuner:
        """
        Create a Ray tuner for the search.
        """
        match self.spec.search.method:
            case "optuna":
                return self._create_optuna_tuner()
            case "hyperopt":
                return self._create_hyperopt_tuner()
            case "random":
                return self._create_random_tuner()
            case _:
                raise ValueError(f"unsupported search method {self.spec.search.method}")

    def _create_random_tuner(self) -> ray.tune.Tuner:
        searcher = ray.tune.search.BasicVariantGenerator(
            random_state=default_rng(self.random_seed.spawn(1)[0])
        )
        return self._create_tuner_for_searcher(searcher)

    def _create_hyperopt_tuner(self) -> ray.tune.Tuner:
        searcher = HyperOptSearch(random_state_seed=int_seed(self.random_seed.spawn(1)[0]))
        return self._create_tuner_for_searcher(searcher)

    def _create_optuna_tuner(self) -> ray.tune.Tuner:
        searcher = OptunaSearch(seed=int_seed(self.random_seed.spawn(1)[0]))
        return self._create_tuner_for_searcher(searcher)

    def _create_tuner_for_searcher(self, searcher) -> ray.tune.Tuner:
        ray_store = self.out_dir / "trial-data"
        scheduler = None
        stopper = None
        cp_config = None
        if self.iterative:
            # FIXME: make this configurable
            min_iter = self.spec.search.min_epochs
            scheduler = ray.tune.schedulers.MedianStoppingRule(
                time_attr="training_iteration",
                grace_period=min_iter,
                min_time_slice=3,
                min_samples_required=3,
            )
            stopper = RelativePlateauStopper(
                metric=self.metric,
                mode=self.mode,
                grace_period=min_iter,
                check_iters=min(min_iter, 3),
                min_improvement=0.005,
            )

            cp_freq = self.spec.search.checkpoint_iters
            self.log.info("will checkpoint every %d iterations", cp_freq)
            cp_config = ray.tune.CheckpointConfig(
                checkpoint_frequency=cp_freq,
                num_to_keep=2,
                # we don't need final model checkpoints
                checkpoint_at_end=False,
            )

        nsamp = self.spec.search.num_search_points()
        space = self.search_space()
        self.log.info("creating tuner for %d samples", nsamp, space=space)
        self.tuner = ray.tune.Tuner(
            self.harness,
            param_space=space,
            tune_config=ray.tune.TuneConfig(
                metric=self.metric,
                mode=self.mode,
                num_samples=nsamp,
                max_concurrent_trials=self.settings.jobs,
                search_alg=searcher,
                scheduler=scheduler,
            ),
            run_config=ray.tune.RunConfig(
                storage_path=ray_store.absolute().as_uri(),
                verbose=None,
                progress_reporter=ProgressReport(self.pipe_name),
                failure_config=ray.tune.FailureConfig(fail_fast=True),
                callbacks=[StatusCallback(self.pipe_name, self.data.train.name)],
                stop=stopper,
                checkpoint_config=cp_config,
            ),
        )
        return self.tuner


@dataclass
class RayTuneResults(TuneResults):
    spec: TuningSpec
    iterative: bool
    results: ray.tune.ResultGrid

    def best_result(self, *, scope: str = "all") -> dict[str, JsonValue]:
        """
        Get the best configuration and its validation metrics.

        Args:
            scope:
                The metric search scope for iterative training.  Set to
                ``"last"`` to use the last iteration instead of the best
                iteration.  See :meth:`ray.tune.ResultGrid.get_best_result` for
                details.
        """
        best = self.results.get_best_result(scope=scope)
        res = best.metrics
        if res is None:
            raise ValueError("best result has no metrics")

        if self.iterative:
            res["config"] = res["config"] | {"epochs": res["training_iteration"]}

        return res

    def best_pipeline(self) -> PipelineConfig:
        """
        Get the (full) configuration for the best pipeline.
        """
        best = self.best_result()
        cfg = self.spec.pipeline
        assert isinstance(cfg, PipelineConfig)
        name = self.spec.component_name
        assert name is not None
        return cfg.merge_component_configs({name: best["config"]})  # type: ignore


def _make_space(space: SearchSpace):
    out = {}
    for name, spec in space.items():
        if isinstance(spec, dict):
            out[name] = _make_space(spec)
        elif spec.type == "int" and spec.scale == "uniform":
            assert isinstance(spec.min, int)
            assert isinstance(spec.max, int)
            out[name] = ray.tune.randint(spec.min, spec.max)
        elif spec.type == "int" and spec.scale == "log":
            assert isinstance(spec.min, int)
            assert isinstance(spec.max, int)
            out[name] = ray.tune.lograndint(spec.min, spec.max, base=spec.base)
        elif spec.type == "float" and spec.scale == "uniform":
            out[name] = ray.tune.uniform(spec.min, spec.max)
        elif spec.type == "float" and spec.scale == "log":
            out[name] = ray.tune.loguniform(spec.min, spec.max, base=spec.base)

    return out
