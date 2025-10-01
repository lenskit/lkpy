# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Hyperparameter searching wrapper.
"""

from pathlib import Path

import numpy as np
import ray.tune
import ray.tune.schedulers
import ray.tune.search
from matplotlib.pylab import default_rng
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch

from lenskit.data import Dataset, ItemListCollection
from lenskit.data.collection._keys import GenericKey
from lenskit.logging import get_logger
from lenskit.parallel import get_parallel_config
from lenskit.pipeline import Pipeline
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
    random_seed: np.random.SeedSequence
    pipeline: Pipeline
    iterative: bool
    job_limit: int | None = None

    data: TTSplit[GenericKey]

    def __init__(
        self,
        spec: TuningSpec,
        spec_path: Path | None = None,
        out_dir: Path | None = None,
        rng: RNGInput = None,
    ):
        if out_dir is None:
            out_dir = Path("lenskit-tune")
        self.out_dir = out_dir

        if isinstance(spec.pipeline, PipelineFile):
            pipe_file = spec.pipeline.file
            if spec_path is not None:
                pipe_file = spec_path.parent / pipe_file
            self.pipeline = Pipeline.load_config(pipe_file)
        else:
            self.pipeline = Pipeline.from_config(spec.pipeline, spec_path)
        self.random_seed = spawn_seed(rng)

        self.log = _log.bind(model=self.pipeline.name)

        if len(self.spec.space) != 1:
            self.log.error("multi-component search is not yet supported")
            raise NotImplementedError()

        comp_name = next(iter(self.spec.space.keys()))
        comp = self.pipeline.node(comp_name)
        match comp:
            case ComponentConstructorNode(constructor=c):
                self.iterative = isinstance(c, type) and issubclass(c, UsesTrainer)
            case _:
                self.log.error("non-class component cannot be searched", component=comp_name)
                raise RuntimeError()

    @property
    def mode(self):
        if self.spec.search.metric == "RMSE":
            return "min"
        else:
            return "max"

    def load_data(self, train: Path, test: Path):
        train_ds = Dataset.load(train)
        test_ilc = ItemListCollection.load_parquet(test)
        self.data = TTSplit(train_ds, test_ilc, name=train_ds.name)

    def search_space(self):
        # we have exactly one
        for space in self.spec.space.values():
            return _make_space(space)

    def setup_harness(self):
        self.log.info("setting up test harness")

        self.job = TuningJobData(
            spec=self.spec,
            pipeline=self.pipeline.config,
            random_seed=self.random_seed.spawn(1)[0],
            data_name=self.data.train.name,
            data_ref=ray.put(self.data),
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
        self.log.info("creating tuner for %d samples", nsamp)
        self.tuner = ray.tune.Tuner(
            self.harness,
            param_space=self.search_space(),
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
                progress_reporter=ProgressReport(self.pipeline.name),
                failure_config=ray.tune.FailureConfig(fail_fast=True),
                callbacks=[StatusCallback(self.pipeline.name, self.data.train.name)],
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
