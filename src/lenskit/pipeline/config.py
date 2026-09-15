# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Old location of pipeline configuration schema.

.. deprecated:: 2026.3.0

    Schema has been moved to :mod:`lenskit.schemas.pipeline`.
"""

from lenskit.schemas.pipeline import (
    UNSET_CODE,
    PipelineComponent,
    PipelineConfig,
    PipelineConfigFragment,
    PipelineHook,
    PipelineHooks,
    PipelineInput,
    PipelineLiteral,
    PipelineMeta,
    PipelineOptions,
    check_name,
    hash_config,
    merge_configs,
)

__all__ = [
    "UNSET_CODE",
    "PipelineComponent",
    "PipelineConfig",
    "PipelineConfigFragment",
    "PipelineHook",
    "PipelineHooks",
    "PipelineInput",
    "PipelineLiteral",
    "PipelineMeta",
    "PipelineOptions",
    "check_name",
    "hash_config",
    "merge_configs",
]
