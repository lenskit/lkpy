"""
Old location of pipeline configuration schema.

.. deprecated:: 2026.3.0

    Schema has been moved to :mod:`lenskit.schemas.pipeline`.
"""

from lenskit.schemas.pipeline import (
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
    "check_name",
    "PipelineHook",
    "PipelineHooks",
    "PipelineOptions",
    "PipelineConfigFragment",
    "PipelineConfig",
    "PipelineMeta",
    "PipelineInput",
    "PipelineComponent",
    "PipelineLiteral",
    "merge_configs",
    "hash_config",
]
