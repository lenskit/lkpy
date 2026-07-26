# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit pipeline abstraction.
"""

from __future__ import annotations

from lenskit.diagnostics import PipelineError, PipelineWarning
from lenskit.lazy import Lazy

from ._builder import PipelineBuilder
from ._cache import PipelineCache
from ._common import RecPipelineBuilder, predict_pipeline, topn_pipeline
from ._diagram import MermaidDiagrammer
from ._hooks import ComponentInputHook
from ._impl import CloneMethod, Pipeline
from ._profiling import PipelineProfiler, ProfileSink
from ._state import PipelineState
from .components import (
    Component,
    ComponentConstructor,
    PipelineFunction,
)
from .config import PipelineConfig
from .nodes import Node

__all__ = [
    "CloneMethod",
    "Component",
    "ComponentConstructor",
    "ComponentInputHook",
    "Lazy",
    "MermaidDiagrammer",
    "Node",
    "Pipeline",
    "PipelineBuilder",
    "PipelineCache",
    "PipelineConfig",
    "PipelineError",
    "PipelineFunction",
    "PipelineProfiler",
    "PipelineState",
    "PipelineWarning",
    "ProfileSink",
    "RecPipelineBuilder",
    "predict_pipeline",
    "topn_pipeline",
]
