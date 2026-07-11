# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Node objects used to represent pipelines internally.  It is very rare
for client code to need to work with the types in this module.

Stability:
    Internal
"""

# pyright: strict
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from types import FunctionType, UnionType
from typing import Any, cast, get_args

from pydantic import BaseModel, JsonValue, TypeAdapter

from ._types import is_union_type, make_importable_path
from .components import (
    Component,
    ComponentConstructor,
    ComponentInput,
    PipelineFunction,
    component_inputs,
)
from .config import PipelineComponent, PipelineInput


class Node[T]:
    """
    Representation of a single node in a :class:`Pipeline`.

    .. versionchanged:: 2026.1

        Removed the ``types`` attribute.

    Stability:
        Caller
    """

    __match_args__ = ("name",)

    name: str
    "The name of this node."

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"

    def to_config(self) -> BaseModel:
        raise TypeError(f"node {self.__class__.__name__} has no config")


class InputNode[T](Node[T]):
    """
    An input node.

    Stability:
        Internal
    """

    type: type[T] | UnionType
    """
    The data type of this input.
    """

    def __init__(self, name: str, *, type: type[T] | UnionType | None = None):
        super().__init__(name)
        if type is None:
            self.type = Any  # type: ignore
        else:
            self.type = type

    def to_config(self) -> PipelineInput:
        if self.type == Any:
            types = None
        elif is_union_type(self.type):  # ty:ignore[invalid-argument-type]
            types = set(get_args(self.type))
        else:
            types = {self.type}

        if types is not None:
            types = {make_importable_path(t) for t in types}

        return PipelineInput(name=self.name, types=types)


class LiteralNode[T](Node[T]):
    """
    A node storing a literal value.

    Stability:
        Internal
    """

    __match_args__ = ("name", "value")
    value: T
    "The value associated with this node"

    def __init__(self, name: str, value: T):
        super().__init__(name)
        self.value = value


class ComponentNode[T](Node[T]):
    """
    A node storing a component.  This is an abstract node class; see subclasses
    :class:`ComponentConstructorNode` and `ComponentInstanceNode`.

    Stability:
        Internal
    """

    @staticmethod
    def create[CFG](
        name: str,
        comp: ComponentConstructor[CFG, T] | Component[T] | PipelineFunction[T],
        config: CFG | Mapping[str, JsonValue] | None = None,
    ) -> ComponentNode[T]:
        if isinstance(comp, Component):
            return ComponentInstanceNode(name, cast(Component[T], comp))
        elif isinstance(comp, ComponentConstructor):
            comp = cast(ComponentConstructor[CFG, T], comp)
            return ComponentConstructorNode(name, comp, comp.validate_config(config))
        elif isinstance(comp, type):
            return ComponentConstructorNode(name, comp, None)  # type: ignore
        else:
            return ComponentInstanceNode(name, comp)

    @property
    @abstractmethod
    def inputs(self) -> dict[str, ComponentInput]:  # pragma: nocover
        raise NotImplementedError()

    # method to refine the return type
    def to_config(self) -> PipelineComponent:
        raise NotImplementedError()


class ComponentConstructorNode[T](ComponentNode[T]):
    __match_args__ = ("name", "constructor", "config")
    constructor: ComponentConstructor[Any, T]
    config: object | None

    def __init__[CFG](
        self, name: str, constructor: ComponentConstructor[CFG, T], config: CFG | None
    ):
        super().__init__(name)
        self.constructor = constructor
        self.config = config

    @property
    def inputs(self):
        return component_inputs(self.constructor)

    def to_config(self) -> PipelineComponent:
        if isinstance(self.constructor, type) and issubclass(self.constructor, Component):
            config = TypeAdapter[Any](self.constructor.config_class()).dump_python(
                self.config, mode="json"
            )
        else:
            config = None

        code = make_importable_path(self.constructor)  # type: ignore

        return PipelineComponent(code=code, config=config)


class ComponentInstanceNode[T](ComponentNode[T]):
    __match_args__ = ("name", "component")

    component: Component[T] | PipelineFunction[T]

    def __init__(
        self,
        name: str,
        component: Component[T] | PipelineFunction[T],
    ):
        super().__init__(name)
        self.component = component

    @property
    def inputs(self):
        return component_inputs(self.component, _warn_level=2)

    def to_config(self) -> PipelineComponent:
        config = None
        if isinstance(self.component, FunctionType):
            ctype = self.component
        else:
            ctype = self.component.__class__
            if isinstance(self.component, Component):
                config = self.component.dump_config()

        code = make_importable_path(ctype)

        return PipelineComponent(code=code, config=config)
