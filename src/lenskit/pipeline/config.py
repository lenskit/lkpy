# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Pydantic models for pipeline configuration and serialization support.
"""

# pyright: strict
from __future__ import annotations

import base64
import pickle
import warnings
from collections import OrderedDict
from hashlib import sha256
from types import FunctionType
from typing import Annotated, Literal, Mapping

from annotated_types import Predicate
from pydantic import AliasChoices, BaseModel, Field, JsonValue, TypeAdapter, ValidationError
from typing_extensions import Any, Self

from ._hooks import HookEntry
from .components import Component
from .nodes import ComponentConstructorNode, ComponentInstanceNode, ComponentNode, InputNode
from .types import type_string


class PipelineHook(BaseModel):
    """
    A single entry in a pipeline's hook configuration.
    """

    function: str
    priority: Annotated[int, Predicate(lambda x: x != 0)] = 1

    @classmethod
    def from_entry(cls, hook: HookEntry[Any]):
        if not isinstance(hook, FunctionType):  # type: ignore
            warnings.warn(f"hook {hook.function} is not a function")
        function = f"{hook.function.__module__}:{hook.function.__qualname__}"
        return cls(function=function, priority=hook.priority)


class PipelineHooks(BaseModel):
    """
    Hook specifications for a pipeline.
    """

    run: dict[str, list[PipelineHook]] = {}


class PipelineOptions(BaseModel, extra="allow"):
    """
    Options used for pipeline assembly, particularly for extending pipelines.
    """

    base: str | None = None
    "Apply this configuration to a base pipeline."

    default_length: int | None = Field(
        default=None, validation_alias=AliasChoices("default_length", "n")
    )
    """
    For top-*N* base pipelines, the default ranking length to configure.
    """

    fallback_predictor: bool | None = None
    """
    For rating prediction pipelines, enable or disable the default fallback
    predictor.
    """


class PipelineConfig(BaseModel):
    """
    Root type for serialized pipeline configuration.  A pipeline config contains
    the full configuration, components, and wiring for the pipeline, but does
    not contain the learned parameters.

    Stability:
        Full
    """

    options: PipelineOptions | None = None
    "Options for assembling the final pipeline."
    meta: PipelineMeta
    "Pipeline metadata."
    inputs: list[PipelineInput] = []
    "Pipeline inputs."
    components: OrderedDict[str, PipelineComponent] = Field(default_factory=OrderedDict)
    "Pipeline components, with their configurations and wiring."
    aliases: dict[str, str] = {}
    "Pipeline node aliases."
    default: str | None = None
    "The default node for running this pipeline."
    literals: dict[str, PipelineLiteral] = {}
    "Literals"
    hooks: PipelineHooks = PipelineHooks()
    "The hooks configured for the pipeline."


class PipelineMeta(BaseModel):
    """
    Pipeline metadata.

    Stability:
        Full
    """

    name: str | None = None
    "The pipeline name."
    version: str | None = None
    "The pipeline version."
    hash: str | None = None
    """
    The pipeline configuration hash.  This is optional, particularly when
    hand-crafting pipeline configuration files.
    """


class PipelineInput(BaseModel):
    """
    Spcification of a pipeline input.

    Stability:
        Full
    """

    name: str
    "The name for this input."
    types: set[str] | None
    "The list of types for this input."

    @classmethod
    def from_node(cls, node: InputNode[Any]) -> Self:
        if node.types is not None:
            types = {type_string(t) for t in node.types}
        else:
            types = None

        return cls(name=node.name, types=types)


class PipelineComponent(BaseModel):
    """
    Specification of a pipeline component.
    """

    code: str = Field(validation_alias=AliasChoices("code", "class"))
    """
    The path to the component's implementation, either a class or a function.
    This is a Python qualified path of the form ``module:name``.
    """

    config: Mapping[str, JsonValue] | None = None
    """
    The component configuration.  If not provided, the component will be created
    with its default constructor parameters.
    """

    inputs: dict[str, str] = {}
    """
    The component's input wirings, mapping input names to node names.  For
    certain meta-nodes, it is specified as a list instead of a dict.
    """

    @classmethod
    def from_node(cls, node: ComponentNode[Any]) -> Self:
        match node:
            case ComponentInstanceNode(_name, comp):
                config = None
                if isinstance(comp, FunctionType):
                    ctype = comp
                else:
                    ctype = comp.__class__
                    if isinstance(comp, Component):
                        config = comp.dump_config()
            case ComponentConstructorNode(_name, ctype, config):
                config = TypeAdapter[Any](ctype.config_class()).dump_python(config, mode="json")
            case _:
                raise TypeError("unexpected node type")

        code = f"{ctype.__module__}:{ctype.__qualname__}"

        return cls(code=code, config=config)


class PipelineLiteral(BaseModel):
    """
    Literal nodes represented in the pipeline.

    Stability:
        Full
    """

    encoding: Literal["json", "base85"]
    value: JsonValue

    @classmethod
    def represent(cls, data: Any) -> Self:
        try:
            return cls(encoding="json", value=data)
        except ValidationError:
            # data is not basic JSON values, so let's pickle it
            dbytes = pickle.dumps(data)
            return cls(encoding="base85", value=base64.b85encode(dbytes).decode("ascii"))

    def decode(self) -> Any:
        "Decode the represented literal."
        match self.encoding:
            case "json":
                return self.value
            case "base85":
                assert isinstance(self.value, str)
                return pickle.loads(base64.b85decode(self.value))


def hash_config(config: BaseModel) -> str:
    """
    Compute the hash of a configuration model.

    Stability:
        Internal
    """
    json = config.model_dump_json(exclude_none=True)
    h = sha256()
    h.update(json.encode())
    return h.hexdigest()
