# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel

from lenskit.pipeline.config import PipelineConfig

SearchSpace: TypeAlias = dict[str, "SearchParam | SearchSpace"]


class SearchConfig(BaseModel):
    """
    Configuration options for the hyperparameter search.
    """

    method: Literal["optuna", "hyperopt", "random"] | None = None
    """
    The search method to use.
    """
    max_points: int = 60
    """
    The maximum number of points to try.
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
    num_cpus: int | Literal["threads", "all-threads"] = "threads"
    """
    The number of CPUs to request from Ray Tune.
    """
    num_gpus: int | float = 0
    """
    The number of GPUs to requrest from Ray Tune.
    """
    checkpoint_iters: int = 2
    """
    The frqeuencey for saving checkpoints.
    """


class TuningSpec(BaseModel, extra="forbid"):
    """
    Data model for hyperparameter tuning specifications.
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
