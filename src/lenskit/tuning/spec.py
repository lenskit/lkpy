# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal

from lenskit.config import lenskit_config, load_config_data
from lenskit.pipeline.config import PipelineConfig

type SearchSpace = dict[str, SearchParam | SearchSpace]
"""
Specification of the (possibly nested) hyperparameter search space for a
component.
"""


class SearchConfig(BaseModel):
    """
    Configuration options for the hyperparameter search.
    """

    method: Literal["tpe", "random", "hyperopt"] | None = None
    """
    The search method to use.
    """
    max_points: int | None = None
    """
    The maximum number of points to try.
    """
    default_points: int = 60
    """
    The default number of search points, if not limited by a maximum configuration.
    """
    max_epochs: int = 100
    """
    The maximum number of epochs to use in iterative training.
    """
    min_epochs: int = 3
    """
    The minimum number of epochs for iterative training.
    """
    metric: str | None = None
    """
    The metric to use.
    """
    list_length: int | None = None
    """
    The length of recommendation lists to use.
    """
    num_cpus: int | Literal["threads", "backend-threads", "all-threads"] = "threads"
    """
    The number of CPUs to request from Ray Tune.
    """
    num_gpus: int | float = 0
    """
    The number of GPUs to requrest from Ray Tune.
    """
    checkpoint_iters: int = 2
    """
    The frequency for saving checkpoints.
    """

    def update_max_points(self, n: int | None):
        """
        Limit the search points to a new maximum, if it exceeds the current maximum.
        """
        if n is not None:
            if self.max_points is None or self.max_points > n:
                self.max_points = n

    def num_search_points(self) -> int:
        """
        Get the number of search points to use.
        """
        cfg = lenskit_config()

        points = self.default_points

        if self.max_points is not None and points > self.max_points:
            points = self.max_points

        if cfg.tuning.max_points is not None and points > cfg.tuning.max_points:
            points = cfg.tuning.max_points

        return points


class TuningSpec(BaseModel, extra="forbid"):
    """
    Data model for hyperparameter tuning specifications.
    """

    @classmethod
    def load(cls, path: Path) -> TuningSpec:
        cfg = load_config_data(path, cls)
        cfg.file_path = path
        return cfg

    file_path: Annotated[Path | None, Field(exclude=True)] = None
    """
    The path to the spec file.
    """

    search: SearchConfig = SearchConfig()
    """
    Options for the hyperparameter search.
    """
    space: dict[str, SearchSpace]
    """
    The search space for tuning.
    """

    pipeline: PipelineFile | PipelineConfig
    """
    The pipeline to tune.
    """

    @property
    def component_name(self) -> str | None:
        """
        Get the name of the tuned component, if the search specifies
        parameters for a single component.
        """
        if len(self.space) == 1:
            for name in self.space.keys():
                return name
        else:
            return None

    def resolve_path(self, path: Path | str) -> Path:
        """
        Resolve a path relative to this specification's file.
        """
        if self.file_path is None:
            return Path(path)
        else:
            return self.file_path.parent / path


class PipelineFile(BaseModel, extra="forbid"):
    """
    Load a pipeline from a file.
    """

    file: Path
    """
    The file from whic to load the pipeline.
    """


class SearchParam(BaseModel):
    type: Literal["int", "float"]
    """
    The type of this parameter.
    """
    min: int | float
    """
    Minimum parameter value.
    """
    max: int | float
    """
    Maximum parameter value.
    """
    scale: Literal["uniform", "log"] = "uniform"
    """
    Search scale for parameter values.
    """
    base: float | None = None
    """
    Base for logarithmic search scales.
    """
