# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, cast

from pydantic import JsonValue
from typing_extensions import Generic, TypeVar

from .components import (
    Component,
    ComponentConstructor,
    PipelineFunction,
    component_inputs,
    component_return_type,
)

# Nodes are (conceptually) immutable data containers, so Node[U] can be assigned
# to Node[T] if U ≼ T.
ND = TypeVar("ND", covariant=True)
CFG = TypeVar("CFG", contravariant=True, bound=object)


class Node(Generic[ND]):
    """
    Representation of a single node in a :class:`Pipeline`.

    Stability:
        Caller
    """

    __match_args__ = ("name",)

    name: str
    "The name of this node."
    types: set[type] | None
    "The set of valid data types of this node, or None for no typechecking."

    def __init__(self, name: str, *, types: set[type] | None = None):
        self.name = name
        self.types = types

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"


class InputNode(Node[ND], Generic[ND]):
    """
    An input node.

    Stability:
        Internal
    """


class LiteralNode(Node[ND], Generic[ND]):
    """
    A node storing a literal value.

    Stability:
        Internal
    """

    __match_args__ = ("name", "value")
    value: ND
    "The value associated with this node"

    def __init__(self, name: str, value: ND, *, types: set[type] | None = None):
        super().__init__(name, types=types)
        self.value = value


class ComponentNode(Node[ND], Generic[ND]):
    """
    A node storing a component.  This is an abstract node class; see subclasses
    :class:`ComponentConstructorNode` and `ComponentInstanceNode`.

    Stability:
        Internal
    """

    def __init__(self, name: str):
        super().__init__(name)

    @staticmethod
    def create(
        name: str,
        comp: ComponentConstructor[CFG, ND] | Component[ND] | PipelineFunction[ND],
        config: CFG | Mapping[str, JsonValue] | None = None,
    ) -> ComponentNode[ND]:
        if isinstance(comp, Component):
            return ComponentInstanceNode(name, cast(Component[ND], comp))
        elif isinstance(comp, ComponentConstructor):
            comp = cast(ComponentConstructor[CFG, ND], comp)
            return ComponentConstructorNode(name, comp, comp.validate_config(config))
        elif isinstance(comp, type):
            return ComponentConstructorNode(name, comp, None)  # type: ignore
        else:
            return ComponentInstanceNode(name, comp)

    @property
    @abstractmethod
    def inputs(self) -> dict[str, type | None]:  # pragma: nocover
        raise NotImplementedError()


class ComponentConstructorNode(ComponentNode[ND], Generic[ND]):
    __match_args__ = ("name", "constructor", "config")
    constructor: ComponentConstructor[Any, ND]
    config: object | None

    def __init__(self, name: str, constructor: ComponentConstructor[CFG, ND], config: CFG | None):
        super().__init__(name)
        self.constructor = constructor
        self.config = config
        if rt := component_return_type(constructor):
            self.types = {rt}

    @property
    def inputs(self):
        return component_inputs(self.constructor)


class ComponentInstanceNode(ComponentNode[ND], Generic[ND]):
    __match_args__ = ("name", "component")

    component: Component[ND] | PipelineFunction[ND]

    def __init__(
        self,
        name: str,
        component: Component[ND] | PipelineFunction[ND],
    ):
        super().__init__(name)
        self.component = component
        if rt := component_return_type(component):
            self.types = {rt}

    @property
    def inputs(self):
        return component_inputs(self.component)
