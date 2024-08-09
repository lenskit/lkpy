# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit pipeline abstraction.
"""

# pyright: strict
from __future__ import annotations

import logging
import warnings
from types import FunctionType
from typing import Literal, cast
from uuid import NAMESPACE_URL, uuid4, uuid5

from typing_extensions import Any, Self, TypeAlias, TypeVar, overload

from lenskit.data import Dataset
from lenskit.pipeline.types import parse_type_string

from .components import (
    AutoConfig,  # noqa: F401 # type: ignore
    Component,
    ConfigurableComponent,
    TrainableComponent,
    instantiate_component,
)
from .config import (
    PipelineComponent,
    PipelineConfig,
    PipelineInput,
    PipelineLiteral,
    PipelineMeta,
    hash_config,
)
from .nodes import ND, ComponentNode, FallbackNode, InputNode, LiteralNode, Node
from .state import PipelineState

__all__ = [
    "Pipeline",
    "PipelineError",
    "PipelineWarning",
    "Node",
    "topn_pipeline",
    "Component",
    "ConfigurableComponent",
    "TrainableComponent",
    "PipelineConfig",
]

_log = logging.getLogger(__name__)

# common type var for quick use
T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")
CloneMethod: TypeAlias = Literal["config", "pipeline-config"]

NAMESPACE_LITERAL_DATA = uuid5(NAMESPACE_URL, "https://ns.lenskit.org/literal-data/")


class PipelineError(Exception):
    """
    Pipeline configuration errors.

    .. note::

        This exception is only to note problems with the pipeline configuration
        and structure (e.g. circular dependencies).  Errors *running* the
        pipeline are raised as-is.
    """


class PipelineWarning(Warning):
    """
    Pipeline configuration and setup warnings.  We also emit warnings to the
    logger in many cases, but this allows critical ones to be visible even if
    the client code has not enabled logging.

    .. note::

        This warning is only to note problems with the pipeline configuration
        and structure (e.g. circular dependencies).  Errors *running* the
        pipeline are raised as-is.
    """


class Pipeline:
    """
    LensKit recommendation pipeline.  This is the core abstraction for using
    LensKit models and other components to produce recommendations in a useful
    way.  It allows you to wire together components in (mostly) abitrary graphs,
    train them on data, and serialize pipelines to disk for use elsewhere.

    If you have a scoring model and just want to generate recommenations with a
    default setup and minimal configuration, see :func:`topn_pipeline`.

    Args:
        name:
            A name for the pipeline.
        version:
            A numeric version for the pipeline.
    """

    name: str | None = None
    version: str | None = None

    _nodes: dict[str, Node[Any]]
    _aliases: dict[str, Node[Any]]
    _defaults: dict[str, Node[Any]]
    _components: dict[str, Component[Any]]
    _hash: str | None = None
    _last: Node[Any] | None = None
    _anon_nodes: set[str]
    "Track generated node names."

    def __init__(self, name: str | None = None, version: str | None = None):
        self.name = name
        self.version = version
        self._nodes = {}
        self._aliases = {}
        self._defaults = {}
        self._components = {}
        self._anon_nodes = set()
        self._clear_caches()

    def meta(self, *, include_hash: bool = True) -> PipelineMeta:
        """
        Get the metadata (name, version, hash, etc.) for this pipeline without
        returning the whole config.

        Args:
            include_hash:
                Whether to include a configuration hash in the metadata.
        """
        meta = PipelineMeta(name=self.name, version=self.version)
        if include_hash:
            meta.hash = self.config_hash()
        return meta

    @property
    def nodes(self) -> list[Node[object]]:
        """
        Get the nodes in the pipeline graph.
        """
        return list(self._nodes.values())

    @overload
    def node(self, node: str, *, missing: Literal["error"] = "error") -> Node[object]: ...
    @overload
    def node(self, node: str, *, missing: Literal["none"] | None) -> Node[object] | None: ...
    @overload
    def node(self, node: Node[T]) -> Node[T]: ...
    def node(
        self, node: str | Node[Any], *, missing: Literal["error", "none"] | None = "error"
    ) -> Node[object] | None:
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
        elif missing == "none" or missing is None:
            return None
        else:
            raise KeyError(f"node {node}")

    def create_input(self, name: str, *types: type[T] | None) -> Node[T]:
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

        node = InputNode[Any](name, types=set((t if t is not None else type(None)) for t in types))
        self._nodes[name] = node
        self._clear_caches()
        return node

    def literal(self, value: T, *, name: str | None = None) -> LiteralNode[T]:
        """
        Create a literal node (a node with a fixed value).

        .. note::
            Literal nodes cannot be serialized witih :meth:`get_config` or
            :meth:`save_config`.
        """
        if name is None:
            name = str(uuid4())
            self._anon_nodes.add(name)
        node = LiteralNode(name, value, types=set([type(value)]))
        self._nodes[name] = node
        self._clear_caches()
        return node

    def set_default(self, name: str, node: Node[Any] | object) -> None:
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

    def get_default(self, name: str) -> Node[Any] | None:
        """
        Get the default wiring for an input name.
        """
        return self._defaults.get(name, None)

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
        self._check_available_name(name)

        node = ComponentNode(name, obj)
        self._nodes[name] = node
        self._components[name] = obj

        self.connect(node, **inputs)

        self._clear_caches()
        self._last = node
        return node

    def replace_component(
        self,
        name: str | Node[ND],
        obj: Component[ND],
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
        of the specified nodes.  If a node is an input node and its value is not
        supplied, it is treated as ``None`` in this case instead of failing the
        run. This method is used for things like filling in optional pipeline
        inputs.  For example, if you want the pipeline to take candidate items
        through an ``items`` input, but look them up from the user's history and
        the training data if ``items`` is not supplied, you would do:

        .. code:: python

            pipe = Pipeline() # allow candidate items to be optionally specified
            items = pipe.create_input('items', list[EntityId], None) # find
            candidates from the training data (optional) lookup_candidates =
            pipe.add_component(
                'select-candidates', UnratedTrainingItemsCandidateSelector(),
                user=history,
            ) # if the client provided items as a pipeline input, use those;
            otherwise # use the candidate selector we just configured.
            candidates = pipe.use_first_of('candidates', items,
            lookup_candidates)

        .. note::

            This method does not distinguish between an input being unspecified
            and explicitly specified as ``None``.

        .. note::

            This method does *not* implement item-level fallbacks, only
            fallbacks at the level of entire results.  That is, you can use it
            to use component A as a fallback for B if B returns ``None``, but it
            will not use B to fill in missing scores for individual items that A
            did not score.  A specific itemwise fallback component is needed for
            such an operation.

        .. note::

            If one of the fallback elements is a component ``A`` that depends on
            another component or input ``B``, and ``B`` is missing or returns
            ``None`` such that ``A`` would usually fail, then ``A`` will be
            skipped and the fallback will move on to the next node. This works
            with arbitrarily-deep transitive chains.

        Args:
            name:
                The name of the node.
            nodes:
                The nodes to try, in order, to satisfy this node.
        """
        node = FallbackNode(name, list(nodes))
        self._nodes[name] = node
        self._clear_caches()
        return node

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

    def component_configs(self) -> dict[str, dict[str, Any]]:
        """
        Get the configurations for the components.  This is the configurations
        only, it does not include pipeline inputs or wiring.
        """
        return {
            name: comp.get_config()
            for (name, comp) in self._components.items()
            if isinstance(comp, ConfigurableComponent)
        }

    def clone(self, how: CloneMethod = "config") -> Pipeline:
        """
        Clone the pipeline, optionally including trained parameters.

        The ``how`` parameter controls how the pipeline is cloned, and what is
        available in the clone pipeline.  It can be one of the following values:

        ``"config"``
            Create fresh component instances using the configurations of the
            components in this pipeline.  When applied to a trained pipeline,
            the clone does **not** have the original's learned parameters. This
            is the default clone method.
        ``"pipeline-config"``
            Round-trip the entire pipeline through :meth:`get_config` and
            :meth:`from_config`.

        Args:
            how:
                The mechanism to use for cloning the pipeline.

        Returns:
            A new pipeline with the same components and wiring, but fresh
            instances created by round-tripping the configuration.
        """
        if how == "pipeline-config":
            cfg = self.get_config()
            return self.from_config(cfg)
        elif how != "config":  # pragma: nocover
            raise NotImplementedError("only 'config' cloning is currently supported")

        clone = Pipeline()

        for node in self.nodes:
            match node:
                case InputNode(name, types=types):
                    if types is None:
                        types = set[type]()
                    clone.create_input(name, *types)
                case LiteralNode(name, value):
                    clone._nodes[name] = LiteralNode(name, value)
                case FallbackNode(name, alts):
                    clone.use_first_of(name, *alts)
                case ComponentNode(name, comp, _inputs, wiring):
                    if isinstance(comp, FunctionType):
                        comp = comp
                    elif isinstance(comp, ConfigurableComponent):
                        comp = comp.__class__.from_config(comp.get_config())  # type: ignore
                    else:
                        comp = comp.__class__()  # type: ignore
                    cn = clone.add_component(node.name, comp)  # type: ignore
                    for wn, wt in wiring.items():
                        clone.connect(cn, **{wn: clone.node(wt)})
                case _:  # pragma: nocover
                    raise RuntimeError(f"invalid node {node}")

        for n, t in self._aliases.items():
            clone.alias(n, t.name)

        for n, t in self._defaults.items():
            clone.set_default(n, clone.node(t.name))

        return clone

    def get_config(self, *, include_hash: bool = True) -> PipelineConfig:
        """
        Get this pipeline's configuration for serialization.  The configuration
        consists of all inputs and components along with their configurations
        and input connections.  It can be serialized to disk (in JSON, YAML, or
        a similar format) to save a pipeline.

        The configuration does **not** include any trained parameter values,
        although the configuration may include things such as paths to
        checkpoints to load such parameters, depending on the design of the
        components in the pipeline.

        .. note::
            Literal nodes (from :meth:`literal`, or literal values wired to
            inputs) cannot be serialized, and this method will fail if they
            are present in the pipeline.
        """
        meta = self.meta(include_hash=False)
        config = PipelineConfig(meta=meta)

        # We map anonymous nodes to hash-based names for stability.  If we ever
        # allow anonymous components, this will need to be adjusted to maintain
        # component ordering, but it works for now since only literals can be
        # anonymous. First handle the anonymous nodes, so we have that mapping:
        remapped: dict[str, str] = {}
        for an in self._anon_nodes:
            node = self._nodes.get(an, None)
            match node:
                case None:
                    # skip nodes that no longer exist
                    continue
                case LiteralNode(name, value):
                    cfg = PipelineLiteral.represent(value)
                    sname = str(uuid5(NAMESPACE_LITERAL_DATA, cfg.model_dump_json()))
                    _log.debug("renamed anonymous node %s to %s", name, sname)
                    remapped[name] = sname
                    config.literals[sname] = cfg
                case _:
                    # the pipeline only generates anonymous literal nodes right now
                    raise RuntimeError(f"unexpected anonymous node {node}")

        # Now we go over all named nodes and add them to the config:
        for node in self.nodes:
            if node.name in remapped:
                continue

            match node:
                case InputNode():
                    config.inputs.append(PipelineInput.from_node(node))
                case LiteralNode(name, value):
                    config.literals[name] = PipelineLiteral.represent(value)
                case ComponentNode(name):
                    config.components[name] = PipelineComponent.from_node(node, remapped)
                case FallbackNode(name, alternatives):
                    config.components[name] = PipelineComponent(
                        code="@use-first-of",
                        inputs=[remapped.get(n.name, n.name) for n in alternatives],
                    )
                case _:  # pragma: nocover
                    raise RuntimeError(f"invalid node {node}")

        config.aliases = {a: t.name for (a, t) in self._aliases.items()}
        config.defaults = {n: t.name for (n, t) in self._defaults.items()}

        if include_hash:
            config.meta.hash = hash_config(config)

        return config

    def config_hash(self) -> str:
        """
        Get a hash of the pipeline's configuration to uniquely identify it for
        logging, version control, or other purposes.

        The hash format and algorithm are not guaranteed, but is stable within a
        LensKit version.  For the same version of LensKit and component code,
        the same configuration will produce the same hash, so long as there are
        no literal nodes.  Literal nodes will *usually* hash consistently, but
        since literals other than basic JSON values are hashed by pickling, hash
        stability depends on the stability of the pickle bytestream.

        In LensKit 2024.1, the configuration hash is computed by computing the
        JSON serialization of the pipeline configuration *without* a hash and
        returning the hex-encoded SHA256 hash of that configuration.
        """
        if self._hash is None:
            # get the config *without* a hash
            cfg = self.get_config(include_hash=False)
            self._hash = hash_config(cfg)
        return self._hash

    @classmethod
    def from_config(cls, config: object) -> Self:
        cfg = PipelineConfig.model_validate(config)
        pipe = cls()
        for inpt in cfg.inputs:
            types: list[type[Any] | None] = []
            if inpt.types is not None:
                types += [parse_type_string(t) for t in inpt.types]
            pipe.create_input(inpt.name, *types)

        # we now add the components and other nodes in multiple passes to ensure
        # that nodes are available before they are wired (since `connect` can
        # introduce out-of-order dependencies).

        # pass 1: add literals
        for name, data in cfg.literals.items():
            pipe.literal(data.decode(), name=name)

        # pass 2: add components
        to_wire: list[PipelineComponent] = []
        for name, comp in cfg.components.items():
            if comp.code.startswith("@"):
                # ignore special nodes in first pass
                continue

            obj = instantiate_component(comp.code, comp.config)
            pipe.add_component(name, obj)
            to_wire.append(comp)

        # pass 3: add meta nodes
        for name, comp in cfg.components.items():
            if comp.code == "@use-first-of":
                if not isinstance(comp.inputs, list):
                    raise PipelineError("@use-first-of must have input list, not dict")
                pipe.use_first_of(name, *[pipe.node(n) for n in comp.inputs])
            elif comp.code.startswith("@"):
                raise PipelineError(f"unsupported meta-component {comp.code}")

        # pass 4: wiring
        for name, comp in cfg.components.items():
            if isinstance(comp.inputs, dict):
                inputs = {n: pipe.node(t) for (n, t) in comp.inputs.items()}
                pipe.connect(name, **inputs)
            elif not comp.code.startswith("@"):
                raise PipelineError(f"component {name} inputs must be dict, not list")

        # pass 5: aliases
        for n, t in cfg.aliases.items():
            pipe.alias(n, t)

        # pass 6: defaults
        for n, t in cfg.defaults.items():
            pipe.set_default(n, pipe.node(t))

        if cfg.meta.hash is not None:
            h2 = pipe.config_hash()
            if h2 != cfg.meta.hash:
                _log.warning("loaded pipeline does not match hash")
                warnings.warn("loaded pipeline config does not match hash", PipelineWarning)

        return pipe

    def train(self, data: Dataset) -> None:
        """
        Trains the pipeline's trainable components (those implementing the
        :class:`TrainableComponent` interface) on some training data.
        """
        for comp in self._components.values():
            _log.debug("testing whether to train %s", comp)
            if isinstance(comp, TrainableComponent):
                comp = cast(TrainableComponent[Any], comp)
                _log.info("training %s", comp)
                comp.train(data)

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
            PipelineError:
                when there is a pipeline configuration error (e.g. a cycle).
            ValueError:
                when one or more required inputs are missing.
            TypeError:
                when one or more required inputs has an incompatible type.
            other:
                exceptions thrown by components are passed through.
        """
        if not nodes:
            if self._last is None:  # pragma: nocover
                raise PipelineError("pipeline has no components")
            nodes = (self._last,)
        state = self.run_all(*nodes, **kwargs)
        results = [state[self.node(n).name] for n in nodes]

        if len(results) > 1:
            return tuple(results)
        else:
            return results[0]

    def run_all(self, *nodes: str | Node[Any], **kwargs: object) -> PipelineState:
        """
        Run all nodes in the pipeline, or all nodes required to fulfill the
        requested node, and return a mapping with the full pipeline state (the
        data attached to each node). This is useful in cases where client code
        needs to be able to inspect the data at arbitrary steps of the pipeline.
        It differs from :meth:`run` in two ways:

        1.  It returns the data from all nodes as a mapping (dictionary-like
            object), not just the specified nodes as a tuple.
        2.  If no nodes are specified, it runs *all* nodes instead of only the
            last node.  This has the consequence of running nodes that are not
            required to fulfill the last node (such scenarios typically result
            from using :meth:`use_first_of`).

        Args:
            nodes:
                The nodes to run, as positional arguments (if no nodes are
                specified, this method runs all nodes).
            kwargs:
                The inputs.

        Returns:
            The full pipeline state, with :attr:`~PipelineState.default` set to
            the last node specified (either the last node in `nodes`, or the
            last node added to the pipeline).
        """
        from .runner import PipelineRunner

        runner = PipelineRunner(self, kwargs)
        node_list = [self.node(n) for n in nodes]
        if not node_list:
            node_list = self.nodes

        last = None
        for node in node_list:
            runner.run(node)
            last = node.name

        return PipelineState(
            runner.state,
            {a: t.name for (a, t) in self._aliases.items()},
            default=last,
            meta=self.meta(),
        )

    def _check_available_name(self, name: str) -> None:
        if name in self._nodes or name in self._aliases:
            raise ValueError(f"pipeline already has node {name}")

    def _check_member_node(self, node: Node[Any]) -> None:
        nw = self._nodes.get(node.name)
        if nw is not node:
            raise PipelineError(f"node {node} not in pipeline")

    def _clear_caches(self):
        if "_hash" in self.__dict__:
            del self._hash


# remaining re-exports
from .common import topn_pipeline  # type: ignore # noqa: E402
