# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np

from lenskit.config import TuneSettings, lenskit_config
from lenskit.data import Dataset, ItemListCollection
from lenskit.logging import get_logger
from lenskit.pipeline import PipelineBuilder
from lenskit.pipeline.nodes import ComponentConstructorNode
from lenskit.random import RNGInput, spawn_seed
from lenskit.splitting import TTSplit
from lenskit.training import UsesTrainer

from .spec import PipelineFile, TuningSpec

_log = get_logger(__name__)


class BasePipelineTuner(ABC):
    """
    Base class for hyperparameter tuning.

    Args:
        spec:
            The tuning specification.
    """

    settings: TuneSettings
    spec: TuningSpec
    out_dir: Path
    pipe_name: str | None
    random_seed: np.random.SeedSequence
    iterative: bool

    data: TTSplit

    def __init__(self, spec: TuningSpec, out_dir: Path | None = None, rng: RNGInput = None):
        cfg = lenskit_config()
        self.settings = cfg.tuning
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

    @property
    def metric(self):
        metric = self.spec.search.metric
        if metric is None:
            raise RuntimeError("no metric specified")
        else:
            return metric

    @property
    def mode(self) -> Literal["min", "max"]:
        if self.spec.search.metric == "RMSE":
            return "min"
        else:
            return "max"

    @abstractmethod
    def run(self) -> TuneResults:
        """
        Run the tuning job, returning the results.
        """


class TuneResults:
    pass
