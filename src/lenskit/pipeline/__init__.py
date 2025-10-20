# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit pipeline abstraction.
"""

from __future__ import annotations

from lenskit.diagnostics import PipelineError, PipelineWarning

from ._hooks import ComponentInputHook
from ._impl import CloneMethod, Pipeline
from ._profiling import PipelineProfiler, ProfileSink
from .builder import PipelineBuilder
from .cache import PipelineCache
from .common import RecPipelineBuilder, predict_pipeline, topn_pipeline
from .components import (
    Component,
    ComponentConstructor,
    PipelineFunction,
)
from .config import PipelineConfig
from .nodes import Node
from .state import PipelineState
from .types import Lazy

__all__ = [
    "Pipeline",
    "PipelineBuilder",
    "PipelineProfiler",
    "ProfileSink",
    "CloneMethod",
    "PipelineError",
    "PipelineWarning",
    "PipelineState",
    "Node",
    "PipelineFunction",
    "PipelineConfig",
    "Lazy",
    "Component",
    "ComponentConstructor",
    "PipelineCache",
    "RecPipelineBuilder",
    "topn_pipeline",
    "predict_pipeline",
    "ComponentInputHook",
]
