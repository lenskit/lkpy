"""
Pydantic models for pipeline configuration and serialization support.
"""

# pyright: strict
from __future__ import annotations

from pydantic import BaseModel
from typing_extensions import Any, Optional, Self

from lenskit.pipeline.types import type_string

from .nodes import InputNode


class PipelineConfig(BaseModel):
    """
    Root type for serialized pipeline configuration.  A pipeline config contains
    the full configuration, components, and wiring for the pipeline, but does
    not contain the
    """

    inputs: list[PipelineInput]


class PipelineInput(BaseModel):
    name: str
    "The name for this input."
    types: Optional[list[str]]
    "The list of types for this input."

    @classmethod
    def from_node(cls, node: InputNode[Any]) -> Self:
        if node.types is not None:
            types = [type_string(t) for t in node.types]
        else:
            types = None

        return cls(name=node.name, types=types)
