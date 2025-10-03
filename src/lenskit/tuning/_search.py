# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Hyperparameter searching wrapper.
"""

from pathlib import Path
from typing import Any

import numpy as np
import ray.tune
import ray.tune.schedulers
import ray.tune.search
from matplotlib.pylab import default_rng
from pydantic import JsonValue
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch

from lenskit.data import Dataset, ItemListCollection
from lenskit.data.collection._keys import GenericKey
from lenskit.logging import get_logger
from lenskit.parallel import get_parallel_config
from lenskit.parallel.ray import ensure_cluster
from lenskit.pipeline import PipelineBuilder
from lenskit.pipeline.nodes import ComponentConstructorNode
from lenskit.random import RNGInput, int_seed, spawn_seed
from lenskit.splitting import TTSplit
from lenskit.training import UsesTrainer

from .iterative import IterativeEval
from .job import TuningJobData
from .reporting import ProgressReport, StatusCallback
from .simple import SimplePointEval
from .spec import PipelineFile, SearchSpace, TuningSpec
from .stopper import RelativePlateauStopper

_log = get_logger(__name__)


class PipelineTuner:
    """
    Set up and run a hyperparameter tuning job for a pipeline.
    """

    spec: TuningSpec
    out_dir: Path
    pipe_name: str | None
    random_seed: np.random.SeedSequence
    iterative: bool
    job_limit: int | None = None

    data: TTSplit[GenericKey]
    harness: Any
    tuner: ray.tune.Tuner
    """
    The Ray tuner that is used for tuning.  Not available until :meth:`setup`
    has been called.
    """
    results: ray.tune.ResultGrid
    """
    Ray tuning results. Only available after :meth:`run` has been called.
    """

    def __init__(
        self,
        spec: TuningSpec,
        out_dir: Path | None = None,
        rng: RNGInput = None,
    ):
        if out_dir is None:
            out_dir = Path("lenskit-tune")
        self.out_dir = out_dir

        if isinstance(spec.pipeline, PipelineFile):
            pipe_file = spec.pipeline.file
            pipe_file = spec.resolve_path(pipe_file)
            pb = PipelineBuilder.load_config(pipe_file)
        else:
            pb = PipelineBuilder.from_config(spec.pipeline, file_path=spec.file_path)

        self.spec = spec.model_copy(deep=True)
        self.spec.pipeline = pb.build_config()
        self.random_seed = spawn_seed(rng)

        self.log = _log.bind(model=pb.name)

        comp_name = self.spec.component_name
        if comp_name is None:
            self.log.error("multi-component search is not yet supported")
            raise NotImplementedError()

        comp = pb.node(comp_name)
        match comp:
            case ComponentConstructorNode(constructor=c):
                self.iterative = isinstance(c, type) and issubclass(c, UsesTrainer)
            case _:
                self.log.error("non-class component cannot be searched", component=comp_name)
                self.log.info("invalid component node: %s", comp)
                raise RuntimeError("attempted to search non-class component")
        self.pipe_name = pb.name

    @property
    def mode(self):
        if self.spec.search.metric == "RMSE":
            return "min"
        else:
            return "max"

    def set_data(
        self, train: Dataset | Path, test: ItemListCollection | Path, *, name: str | None = None
    ):
        """
        Set the data to be used for tuning.
        """
        if isinstance(train, Path):
            train = Dataset.load(train)
        if isinstance(test, Path):
            test = ItemListCollection.load_parquet(test)

        self.data = TTSplit(train, test, name=name or train.name)

    def setup(self):
        """
        Set up to run the trainer.  After this method completes, the
        :attr:`tuner` is ready.
        """
        ensure_cluster()
        self.setup_harness()
        self.tuner = self.create_tuner()
        self.out_dir.mkdir(exist_ok=True, parents=True)

    def run(self) -> ray.tune.ResultGrid:
        """
        Run the tuning job.

        Saves the results in :attr:`results`, and also returns them.
        """
        if not hasattr(self, "tuner"):
            self.setup()

        self.log.info("starting hyperparameter search")
        self.results = self.tuner.fit()
        self.log.info("finished hyperparameter search")

        return self.results

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

        if "training_iteration" in res:
            res["config"] = res["config"] | {"epochs": res["training_iteration"]}

        return res

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

        self.log.info(
            "setting up parallel tuner",
            cpus=paracfg.total_threads,
        )

        match self.spec.search.num_cpus:
            case "threads":
                tune_cpus = paracfg.threads
            case "all-threads":
                tune_cpus = paracfg.total_threads
            case int(n) if n > 0:
                tune_cpus = n
            case _:
                raise ValueError(f"invalid CPU count {self.spec.search.num_cpus}")

        self.harness = ray.tune.with_resources(
            harness, {"CPU": tune_cpus, "GPU": self.spec.search.num_gpus}
        )

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
        ray_store = self.out_dir / "tuning-state"
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

        nsamp = self.spec.search.max_points
        space = self.search_space()
        self.log.info("creating tuner for %d samples", nsamp, space=space)
        self.tuner = ray.tune.Tuner(
            self.harness,
            param_space=space,
            tune_config=ray.tune.TuneConfig(
                metric=self.metric,
                mode=self.mode,
                num_samples=nsamp,
                max_concurrent_trials=self.job_limit,
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
