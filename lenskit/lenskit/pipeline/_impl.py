# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

import logging
import warnings
from inspect import Signature, signature
from typing import Callable, cast
from uuid import uuid4

from typing_extensions import Any, Generic, LiteralString, TypeVar, overload

from lenskit.data import Dataset
from lenskit.pipeline.types import TypecheckWarning, is_compatible_data

from .components import Component

_log = logging.getLogger(__name__)

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


class Node(Generic[ND]):
    """
    Representation of a single node in a :class:`Pipeline`.
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
    """


class LiteralNode(Node[ND], Generic[ND]):
    __match_args__ = ("name", "value")
    value: ND
    "The value associated with this node"

    def __init__(self, name: str, value: ND, *, types: set[type] | None = None):
        super().__init__(name, types=types)
        self.value = value


class ComponentNode(Node[ND], Generic[ND]):
    __match_args__ = ("name", "component", "inputs", "connections")

    component: Component[ND]
    "The component associated with this node"

    inputs: dict[str, type | None]
    "The component's inputs."

    connections: dict[str, str]
    "The component's input connections."

    def __init__(self, name: str, component: Component[ND]):
        super().__init__(name)
        self.component = component
        self.connections = {}

        sig = signature(component)
        if sig.return_annotation == Signature.empty:
            warnings.warn(
                f"component {component} has no return type annotation", TypecheckWarning, 2
            )
        else:
            self.types = set([sig.return_annotation])

        self.inputs = {
            param.name: None if param.annotation == Signature.empty else param.annotation
            for param in sig.parameters.values()
        }


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
    _aliases: dict[str, Node[Any]]
    _defaults: dict[str, Node[Any] | Any]
    _components: dict[str, Component[Any]]

    def __init__(self):
        self._nodes = {}
        self._aliases = {}
        self._defaults = {}
        self._components = {}
        self._clear_caches()

    @property
    def nodes(self) -> list[Node[object]]:
        """
        Get the nodes in the pipeline graph.
        """
        return list(self._nodes.values())

    @overload
    def node(self, node: str) -> Node[object]: ...
    @overload
    def node(self, node: Node[T]) -> Node[T]: ...
    def node(self, node: str | Node[Any]) -> Node[object]:
        """
        Get the pipeline node with the specified name.  If passed a node, it
        returns the node or fails if the node is not a member of the pipeline.

        Args:
            node:
                The name of the pipeline node to look up, or a node to check for
                membership.

        Returns:
            The pipeline node, if it exists.

        Raises:
            KeyError:
                The specified node does not exist.
        """
        if isinstance(node, Node):
            self._check_member_node(node)
            return node
        elif node in self._aliases:
            return self._aliases[node]
        elif node in self._nodes:
            return self._nodes[node]
        else:
            raise KeyError(f"node {node}")

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
        self._check_available_name(name)

        node = InputNode[Any](name, types=set(types))
        self._nodes[name] = node
        self._clear_caches()
        return node

    def literal(self, value: T) -> LiteralNode[T]:
        name = str(uuid4())
        node = LiteralNode(name, value, types=set([type(value)]))
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
        if not isinstance(node, Node):
            node = self.literal(node)
        self._defaults[name] = node
        self._clear_caches()

    def alias(self, alias: str, node: Node[Any] | str) -> None:
        """
        Create an alias for a node.  After aliasing, the node can be retrieved
        from :meth:`node` using either its original name or its alias.

        Args:
            alias:
                The alias to add to the node.
            node:
                The node (or node name) to alias.

        Raises:
            ValueError:
                if the alias is already used as an alias or node name.
        """
        node = self.node(node)
        self._check_available_name(alias)
        self._aliases[alias] = node
        self._clear_caches()

    def add_component(
        self, name: str, obj: Component[ND] | Callable[..., ND], **inputs: Node[Any] | object
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
        self._check_available_name(name)

        node = ComponentNode(name, obj)
        self._nodes[name] = node
        self._components[name] = obj

        self.connect(node, **inputs)

        self._clear_caches()
        return node

    def replace_component(
        self,
        name: str | Node[ND],
        obj: Component[ND] | Callable[..., ND],
        **inputs: Node[Any] | object,
    ) -> Node[ND]:
        """
        Replace a component in the graph.  The new component must have a type
        that is compatible with the old component.  The old component's input
        connections will be replaced (as the new component may have different
        inputs), but any connections that use the old component to supply an
        input will use the new component instead.
        """
        if isinstance(name, Node):
            name = name.name

        node = ComponentNode(name, obj)
        self._nodes[name] = node
        self._components[name] = obj

        self.connect(node, **inputs)

        self._clear_caches()
        return node

    def use_first_of(self, name: str, *nodes: Node[T | None]) -> Node[T]:
        """
        Create a new node whose value is the first defined (not ``None``) value
        of the specified nodes.  This is used for things like filling in optional
        pipeline inputs.  For example, if you want the pipeline to take candidate
        items through an ``items`` input, but look them up from the user's history
        and the training data if ``items`` is not supplied, you would do:

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

        .. note::

            This method does *not* implement item-level fallbacks, only fallbacks at
            the level of entire results.  That is, you can use it to use component A
            as a fallback for B if B returns ``None``, but it will not use B to fill
            in missing scores for individual items that A did not score.  A specific
            itemwise fallback component is needed for such an operation.
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
        if isinstance(obj, Node):
            node = obj
        else:
            node = self.node(obj)
        if not isinstance(node, ComponentNode):
            raise TypeError(f"only component nodes can be wired, not {node}")

        for k, n in inputs.items():
            if isinstance(n, Node):
                n = cast(Node[Any], n)
                self._check_member_node(n)
                node.connections[k] = n.name
            else:
                lit = self.literal(n)
                node.connections[k] = lit.name

        self._clear_caches()

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
    def run(self, *nodes: str | Node[Any], **kwargs: object) -> object:
        """
        Run the pipeline and obtain the return value(s) of one or more of its
        components.  See :ref:`pipeline-execution` for details of the pipeline
        execution model.

        .. todo::
            Add cycle detection.

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
        state: dict[str, Any] = {}

        ret: list[Node[Any]] | Node[Any] = [self.node(n) for n in nodes]
        _log.debug(
            "starting run of pipeline with %d nodes, want %s",
            len(self._nodes),
            [n.name for n in ret],
        )
        if not ret:
            ret = [self._last_node()]

        # set up a stack of nodes to look at
        # we traverse the graph with this
        needed = list(reversed(ret))

        # the main loop — keep resolving pipeline nodes until we're done
        while needed:
            node = needed[-1]
            if node.name in state:
                # the node is computed, we're done
                needed.pop()
                continue

            match node:
                case LiteralNode(name, value):
                    # literal nodes are ready to put on the state
                    state[name] = value
                    needed.pop()
                case ComponentNode(name, comp, inputs, wiring):
                    # check that (1) the node is fully wired, and (2) its inputs are all computed
                    ready = True
                    for k in inputs.keys():
                        if k in wiring:
                            wired = wiring[k]
                        elif k in self._defaults:
                            wired = self._defaults[k]
                        else:
                            raise RuntimeError(f"input {k} for {node} not connected")
                        wired = self.node(wired)

                        if wired.name not in state:
                            # input value not available, queue it up
                            ready = False
                            # it is fine to queue the same node twice — it will
                            # be quickly skipped the second time
                            needed.append(wired)

                    if ready:
                        _log.debug("running %s (%s)", node, comp)
                        # if the node is ready to compute (all inputs in state), we run it.
                        args = {}
                        for n in inputs.keys():
                            if n in wiring:
                                args[n] = state[wiring[n]]
                            elif n in self._defaults:
                                args[n] = state[self._defaults[n].name]
                            else:  # pragma: nocover
                                raise AssertionError("missing input not caught earlier")
                        state[name] = comp(**args)
                        needed.pop()

                    # fallthrough: the node is not ready, and we have pushed its
                    # inputs onto the stack.  The inputs may be re-pushed, so this
                    # will never be the last node on the stack at this point

                case InputNode(name, types=types):
                    try:
                        val = kwargs[name]
                    except KeyError:
                        raise RuntimeError(f"input {name} not specified")

                    if types and not is_compatible_data(val, *types):
                        raise TypeError(
                            f"invalid data for input {name} (expected {types}, got {type(val)})"
                        )
                    state[name] = val
                    needed.pop()

                case _:
                    raise RuntimeError(f"invalid node {node}")

        if len(ret) > 1:
            return tuple(state[r.name] for r in ret)
        else:
            return state[ret[0].name]

    def _last_node(self) -> Node[object]:
        if not self._nodes:
            raise RuntimeError("pipeline is empty")
        return list(self._nodes.values())[-1]

    def _check_available_name(self, name: str) -> None:
        if name in self._nodes or name in self._aliases:
            raise ValueError(f"pipeline already has node {name}")

    def _check_member_node(self, node: Node[Any]) -> None:
        nw = self._nodes.get(node.name)
        if nw is not node:
            raise RuntimeError(f"node {node} not in pipeline")

    def _clear_caches(self):
        pass
