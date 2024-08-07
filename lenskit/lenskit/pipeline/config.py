"""
Pydantic models for pipeline configuration and serialization support.
"""

# pyright: strict
from __future__ import annotations

from types import FunctionType

from pydantic import BaseModel, Field
from typing_extensions import Any, Optional, Self

from .components import ConfigurableComponent
from .nodes import ComponentNode, InputNode
from .types import type_string


class PipelineConfig(BaseModel):
    """
    Root type for serialized pipeline configuration.  A pipeline config contains
    the full configuration, components, and wiring for the pipeline, but does
    not contain the
    """

    inputs: list[PipelineInput]
    components: dict[str, PipelineComponent] = Field(default_factory=dict)


class PipelineInput(BaseModel):
    name: str
    "The name for this input."
    types: Optional[set[str]]
    "The list of types for this input."

    @classmethod
    def from_node(cls, node: InputNode[Any]) -> Self:
        if node.types is not None:
            types = {type_string(t) for t in node.types}
        else:
            types = None

        return cls(name=node.name, types=types)


class PipelineComponent(BaseModel):
    code: str
    """
    The path to the component's implementation, either a class or a function.
    This is a Python qualified path of the form ``module:name``.
    """

    config: dict[str, object] | None = Field(default=None)
    """
    The component configuration.  If not provided, the component will be created
    with its default constructor parameters.
    """

    inputs: dict[str, str] = Field(default_factory=dict)
    """
    The component's input wirings, mapping input names to node names.
    """

    @classmethod
    def from_node(cls, node: ComponentNode[Any]) -> Self:
        comp = node.component
        if isinstance(comp, FunctionType):
            ctype = comp
        else:
            ctype = comp.__class__

        code = f"{ctype.__module__}:{ctype.__qualname__}"

        config = comp.get_config() if isinstance(comp, ConfigurableComponent) else None

        return cls(code=code, config=config, inputs=node.connections)
