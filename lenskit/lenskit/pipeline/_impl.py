# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    LiteralString,
    TypeVar,
    overload,
)

from lenskit.data import Dataset

from .components import Component

# Nodes are (conceptually) immutable data containers, so Node[U] can be assigned
# to Node[T] if U ≼ T.
ND = TypeVar("ND", covariant=True)
# common type var for quick use
T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")


@dataclass
class Node(Generic[ND]):
    """
    Representation of a single node in a :class:`Pipeline`.
    """

    name: str
    "The name of this node."
    types: set[type] | None = None
    "The set of valid data types of this node."


class Pipeline:
    """
    LensKit recommendation pipeline.  This is the core abstraction for using
    LensKit models and other components to produce recommendations in a useful
    way.  It allows you to wire together components in (mostly) abitrary graphs,
    train them on data, and serialize pipelines to disk for use elsewhere.

    If you have a scoring model and just want to generate recommenations with a
    default setup and minimal configuration, see :func:`topn_pipeline`.
    """

    _nodes: dict[str, Node[Any]]
    _defaults: dict[str, Node[Any] | Any]
    _components: dict[str, Component[Any]]

    def __init__(self):
        self._nodes = {}
        self._defaults = {}
        self._components = {}

    @property
    def nodes(self) -> list[Node[object]]:
        """
        Get the nodes in the pipeline graph.
        """
        return list(self._nodes.values())

    def node(self, name: str) -> Node[object]:
        """
        Get the pipeline node with the specified name.

        Args:
            name:
                The name of the pipeline node to look up.

        Returns:
            The pipeline node, if it exists.

        Raises:
            KeyError:
                The specified node does not exist.
        """
        return self._nodes[name]

    def create_input(self, name: LiteralString, *types: type[T]) -> Node[T]:
        """
        Create an input node for the pipeline.  Pipelines expect their inputs to
        be provided when they are run.

        Args:
            name:
                The name of the input.  The name must be unique in the pipeline
                (among both components and inputs).
            types:
                The allowable types of the input; input data can be of any
                specified type.  If ``None`` is among the allowed types, the
                input can be omitted.

        Returns:
            A pipeline node representing this input.

        Raises:
            ValueError:
                a node with the specified ``name`` already exists.
        """
        if name in self._nodes:
            raise ValueError(f"pipeline already has node “{name}”")

        node = Node[T](name, set(types))
        self._nodes[name] = node
        return node

    def set_default(self, name: LiteralString, node: Node[Any] | object) -> None:
        """
        Set the default wiring for a component input.  Components that declare
        an input parameter with the specified ``name`` but no configured input
        will be wired to this node.

        This is intended to be used for things like wiring up `user` parameters
        to semi-automatically receive the target user's identity and history.

        Args:
            name:
                The name of the parameter to set a default for.
            node:
                The node or literal value to wire to this parameter.
        """

    def add_component(
        self, name: str, obj: Component[ND], **inputs: Node[Any] | object
    ) -> Node[ND]:
        """
        Add a component and connect it into the graph.

        Args:
            name:
                The name of the component in the pipeline.  The name must be
                unique in the pipeline (among both components and inputs).
            obj:
                The component itself.
            inputs:
                The component's input wiring.  See :ref:`pipeline-connections`
                for details.

        Returns:
            The node representing this component in the pipeline.
        """
        return Node(name)

    def use_first_of(self, name: str, *nodes: Node[T | None]) -> Node[T]:
        """
        Create a new node whose value is the first defined (not ``None``) value
        of the specified nodes.  This is used for things like filling in optional
        pipeline inputs.  For example, if you want the pipeline to take candidate
        items through an `items` input, but look them up from the user's history
        and the training data if `items` is not supplied, you would do:

        .. code:: python

            pipe = Pipeline()
            # allow candidate items to be optionally specified
            items = pipe.create_input('items', list[EntityId], None)
            # find candidates from the training data (optional)
            lookup_candidates = pipe.add_component(
                'select-candidates',
                UnratedTrainingItemsCandidateSelector(),
                user=history,
            )
            # if the client provided items as a pipeline input, use those; otherwise
            # use the candidate selector we just configured.
            candidates = pipe.use_first_of('candidates', items, lookup_candidates)
        """
        raise NotImplementedError()

    def connect(self, obj: str | Node[Any], **inputs: Node[Any] | str | object):
        """
        Provide additional input connections for a component that has already
        been added.  See :ref:`pipeline-connections` for details.

        Args:
            obj:
                The name or node of the component to wire.
            inputs:
                The component's input wiring.  For each keyword argument in the
                component's function signature, that argument can be provided
                here with an input that the pipeline will provide to that
                argument of the component when the pipeline is run.
        """

    def train(self, data: Dataset) -> None:
        """
        Trains the pipeline's trainable components (those implementing the
        :class:`TrainableComponent` interface) on some training data.
        """

    @overload
    def run(self, /, **kwargs: object) -> object: ...
    @overload
    def run(self, node: str, /, **kwargs: object) -> object: ...
    @overload
    def run(self, n1: str, n2: str, /, *nrest: str, **kwargs: object) -> tuple[object]: ...
    @overload
    def run(self, node: Node[T], /, **kwargs: object) -> T: ...
    @overload
    def run(self, n1: Node[T1], n2: Node[T2], /, **kwargs: object) -> tuple[T1, T2]: ...
    @overload
    def run(
        self, n1: Node[T1], n2: Node[T2], n3: Node[T3], /, **kwargs: object
    ) -> tuple[T1, T2, T3]: ...
    @overload
    def run(
        self, n1: Node[T1], n2: Node[T2], n3: Node[T3], n4: Node[T4], /, **kwargs: object
    ) -> tuple[T1, T2, T3, T4]: ...
    @overload
    def run(
        self,
        n1: Node[T1],
        n2: Node[T2],
        n3: Node[T3],
        n4: Node[T4],
        n5: Node[T5],
        /,
        **kwargs: object,
    ) -> tuple[T1, T2, T3, T4, T5]: ...
    def run(self, *nodes: str | Node[Any] | None, **kwargs: object) -> object:
        """
        Run the pipeline and obtain the return value(s) of one or more of its
        components.  See :ref:`pipeline-execution` for details of the pipeline
        execution model.

        Args:
            nodes:
                The component(s) to run.
            kwargs:
                The pipeline's inputs, as defined with :meth:`create_input`.

        Returns:
            The pipeline result.  If zero or one nodes are specified, the result
            is returned as-is. If multiple nodes are specified, their results
            are returned in a tuple.

        Raises:
            ValueError:
                when one or more required inputs are missing.
            TypeError:
                when one or more required inputs has an incompatible type.
            other:
                exceptions thrown by components are passed through.
        """
        pass
