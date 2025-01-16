"""
LensKit pipeline builder.
"""

# pyright: strict
from __future__ import annotations

import typing
import warnings
from types import UnionType
from uuid import NAMESPACE_URL, uuid4, uuid5

from typing_extensions import Any, Literal, Self, TypeVar, cast, overload

from lenskit.diagnostics import PipelineError, PipelineWarning
from lenskit.logging import get_logger

from . import config
from ._impl import Pipeline
from .components import (  # type: ignore # noqa: F401
    Component,
    ComponentConstructor,
    PipelineFunction,
    fallback_on_none,
    instantiate_component,
)
from .config import PipelineConfig
from .nodes import (
    ND,
    ComponentConstructorNode,
    ComponentInstanceNode,
    ComponentNode,
    InputNode,
    LiteralNode,
    Node,
)
from .types import TypecheckWarning, parse_type_string

_log = get_logger(__name__)

CFG = TypeVar("CFG")
# common type var for quick use
T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

NAMESPACE_LITERAL_DATA = uuid5(NAMESPACE_URL, "https://ns.lenskit.org/literal-data/")


class PipelineBuilder:
    """
    Builder for LensKit recommendation pipelines.  :ref:`Pipelines <pipeline>`
    are the core abstraction for using LensKit models and other components to
    produce recommendations in a useful way.  They allow you to wire together
    components in (mostly) abitrary graphs, train them on data, and serialize
    the resulting pipelines to disk for use elsewhere.

    The builder configures and builds pipelines that can then be run. If you
    have a scoring model and just want to generate recommenations with a default
    setup and minimal configuration, see :func:`~lenskit.pipeline.topn_pipeline`
    or :class:`~lenskit.pipeline.RecPipelineBuilder`.

    Args:
        name:
            A name for the pipeline.
        version:
            A numeric version for the pipeline.

    Stability:
        Caller
    """

    name: str | None = None
    """
    The pipeline name.
    """
    version: str | None = None
    """
    The pipeline version string.
    """

    _nodes: dict[str, Node[Any]]
    _aliases: dict[str, Node[Any]]
    _default_connections: dict[str, str]
    _components: dict[str, PipelineFunction[Any] | Component[Any]]
    _default: str | None = None
    _anon_nodes: set[str]
    "Track generated node names."

    def __init__(self, name: str | None = None, version: str | None = None):
        self.name = name
        self.version = version
        self._nodes = {}
        self._aliases = {}
        self._default_connections = {}
        self._components = {}
        self._anon_nodes = set()

    def meta(self, *, include_hash: bool = True) -> config.PipelineMeta:
        """
        Get the metadata (name, version, hash, etc.) for this pipeline without
        returning the whole config.

        Args:
            include_hash:
                Whether to include a configuration hash in the metadata.
        """
        meta = config.PipelineMeta(name=self.name, version=self.version)
        if include_hash:
            meta.hash = self.config_hash()
        return meta

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

    def create_input(self, name: str, *types: type[T] | UnionType | None) -> Node[T]:
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

        rts: set[type[T | None]] = set()
        for t in types:
            if t is None:
                rts.add(type(None))
            elif isinstance(t, UnionType):
                rts |= set(typing.get_args(t))
            else:
                rts.add(t)

        node = InputNode[Any](name, types=rts)
        self._nodes[name] = node
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
        return node

    def default_connection(self, name: str, node: Node[Any] | object) -> None:
        """
        Set the default wiring for a component input.  Components that declare
        an input parameter with the specified ``name`` but no configured input
        will be wired to this node.

        This is intended to be used for things like wiring up `user` parameters
        to semi-automatically receive the target user's identity and history.

        .. important::

            Defaults are a feature of the builder only, and are resolved in
            :meth:`build`.  They are not included in serialized configuration or
            resulting pipeline.

        Args:
            name:
                The name of the parameter to set a default for.
            node:
                The node or literal value to wire to this parameter.
        """
        if not isinstance(node, Node):
            node = self.literal(node)
        self._default_connections[name] = node.name

    def default_component(self, node: str | Node[Any]) -> None:
        """
        Set the default node for the pipeline.  If :meth:`Pipeline.run` is
        called without a node, then it will run this node (and all of its
        dependencies).
        """
        self._default = node.name if isinstance(node, Node) else node

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

    @overload
    def add_component(
        self,
        name: str,
        cls: ComponentConstructor[CFG, ND],
        config: CFG = None,
        /,
        **inputs: Node[Any],
    ) -> Node[ND]: ...
    @overload
    def add_component(
        self,
        name: str,
        instance: Component[ND] | PipelineFunction[ND],
        /,
        **inputs: Node[Any] | object,
    ) -> Node[ND]: ...
    def add_component(
        self,
        name: str,
        comp: ComponentConstructor[CFG, ND] | Component[ND] | PipelineFunction[ND],
        config: CFG | None = None,
        /,
        **inputs: Node[Any] | object,
    ) -> Node[ND]:
        """
        Add a component and connect it into the graph.

        Args:
            name:
                The name of the component in the pipeline.  The name must be
                unique in the pipeline (among both components and inputs).
            cls:
                A component class.
            config:
                The configuration object for the component class.
            instance:
                A raw function or pre-instantiated component.
            inputs:
                The component's input wiring.  See :ref:`pipeline-connections`
                for details.

        Returns:
            The node representing this component in the pipeline.
        """
        self._check_available_name(name)

        node = ComponentNode[ND].create(name, comp, config)
        if node.types is None:
            warnings.warn(f"cannot determine return type of component {comp}", TypecheckWarning)
        self._nodes[name] = node

        self.connect(node, **inputs)

        return node

    @overload
    def replace_component(
        self,
        name: str | Node[ND],
        cls: ComponentConstructor[CFG, ND],
        config: CFG = None,
        /,
        **inputs: Node[Any],
    ) -> Node[ND]: ...
    @overload
    def replace_component(
        self,
        name: str | Node[ND],
        instance: Component[ND] | PipelineFunction[ND],
        /,
        **inputs: Node[Any] | object,
    ) -> Node[ND]: ...
    def replace_component(
        self,
        name: str | Node[ND],
        comp: ComponentConstructor[CFG, ND] | Component[ND] | PipelineFunction[ND],
        config: CFG | None = None,
        /,
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

        node = ComponentNode[ND].create(name, comp, config)
        self._nodes[name] = node

        self.connect(node, **inputs)

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

    def clone(self) -> PipelineBuilder:
        """
        Clone the pipeline builder.  The resulting builder starts as a copy of
        this builder, and any subsequent modifications only the copy to which
        they are applied.
        """

        clone = PipelineBuilder()

        for node in self.nodes():
            match node:
                case InputNode(name, types=types):
                    if types is None:
                        types = set[type]()
                    clone.create_input(name, *types)
                case LiteralNode(name, value):
                    clone._nodes[name] = LiteralNode(name, value)
                case ComponentConstructorNode(name, comp, config):
                    cn = clone.add_component(name, comp, config)
                case ComponentInstanceNode(name, comp):
                    cn = clone.add_component(name, comp)
                case _:  # pragma: nocover
                    raise RuntimeError(f"invalid node {node}")

        for n, t in self._aliases.items():
            clone.alias(n, t.name)

        for node in self.nodes():
            match node:
                case ComponentNode(name, connections=wiring):
                    cn = clone.node(name)
                    clone.connect(cn, **{wn: clone.node(wt) for (wn, wt) in wiring.items()})
                case _:
                    pass

        for n, t in self._default_connections.items():
            clone.default_connection(n, clone.node(t))

        return clone

    def build_config(self, *, include_hash: bool = True) -> PipelineConfig:
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
        cfg = PipelineConfig(meta=meta)

        # FIXME: don't mutate
        for node in self._nodes.values():
            if isinstance(node, ComponentNode):
                for iname in node.inputs.keys():
                    if iname not in node.connections and iname in self._default_connections:
                        node.connections[iname] = self._default_connections[iname]

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
                    lit = config.PipelineLiteral.represent(value)
                    sname = str(uuid5(NAMESPACE_LITERAL_DATA, lit.model_dump_json()))
                    _log.debug("renamed anonymous node %s to %s", name, sname)
                    remapped[name] = sname
                    cfg.literals[sname] = lit
                case _:
                    # the pipeline only generates anonymous literal nodes right now
                    raise RuntimeError(f"unexpected anonymous node {node}")

        # Now we go over all named nodes and add them to the config:
        for node in self.nodes():
            if node.name in remapped:
                continue

            match node:
                case InputNode():
                    cfg.inputs.append(config.PipelineInput.from_node(node))
                case LiteralNode(name, value):
                    cfg.literals[name] = config.PipelineLiteral.represent(value)
                case ComponentNode(name):
                    cfg.components[name] = config.PipelineComponent.from_node(node, remapped)
                case _:  # pragma: nocover
                    raise RuntimeError(f"invalid node {node}")

        cfg.aliases = {a: t.name for (a, t) in self._aliases.items()}

        if self._default:
            cfg.default = self._default

        if include_hash:
            cfg.meta.hash = config.hash_config(cfg)

        return cfg

    def config_hash(self) -> str:
        """
        Get a hash of the pipeline's configuration to uniquely identify it for
        logging, version control, or other purposes.

        The hash format and algorithm are not guaranteed, but hashes are stable
        within a LensKit version.  For the same version of LensKit and component
        code, the same configuration will produce the same hash, so long as
        there are no literal nodes.  Literal nodes will *usually* hash
        consistently, but since literals other than basic JSON values are hashed
        by pickling, hash stability depends on the stability of the pickle
        bytestream.

        In LensKit 2025.1, the configuration hash is computed by computing the
        JSON serialization of the pipeline configuration *without* a hash and
        returning the hex-encoded SHA256 hash of that configuration.
        """

        cfg = self.build_config(include_hash=False)
        return config.hash_config(cfg)

    @classmethod
    def from_config(cls, config: object) -> Self:
        """
        Reconstruct a pipeline builder from a serialized configuration.

        Args:
            config:
                The configuration object, as loaded from JSON, TOML, YAML, or
                similar. Will be validated into a :class:`PipelineConfig`.
        Returns:
            The configured (but not trained) pipeline.
        Raises:
            PipelineError:
                If there is a configuration error reconstructing the pipeline.
        Warns:
            PipelineWarning:
                If the configuration is funny but usable; for example, the
                configuration includes a hash but the constructed pipeline does
                not have a matching hash.
        """
        cfg = PipelineConfig.model_validate(config)
        builder = cls()
        for inpt in cfg.inputs:
            types: list[type[Any] | None] = []
            if inpt.types is not None:
                types += [parse_type_string(t) for t in inpt.types]
            builder.create_input(inpt.name, *types)

        # we now add the components and other nodes in multiple passes to ensure
        # that nodes are available before they are wired (since `connect` can
        # introduce out-of-order dependencies).

        # pass 1: add literals
        for name, data in cfg.literals.items():
            builder.literal(data.decode(), name=name)

        # pass 2: add components
        to_wire: list[config.PipelineComponent] = []
        for name, comp in cfg.components.items():
            if comp.code.startswith("@"):
                # ignore special nodes in first pass
                continue

            obj = instantiate_component(comp.code, comp.config)
            builder.add_component(name, obj)
            to_wire.append(comp)

        # pass 3: wiring
        for name, comp in cfg.components.items():
            if isinstance(comp.inputs, dict):
                inputs = {n: builder.node(t) for (n, t) in comp.inputs.items()}
                builder.connect(name, **inputs)
            elif not comp.code.startswith("@"):
                raise PipelineError(f"component {name} inputs must be dict, not list")

        # pass 4: aliases
        for n, t in cfg.aliases.items():
            builder.alias(n, t)

        builder._default = cfg.default

        if cfg.meta.hash is not None:
            h2 = builder.config_hash()
            if h2 != cfg.meta.hash:
                _log.warning("loaded pipeline does not match hash")
                warnings.warn("loaded pipeline config does not match hash", PipelineWarning)

        return builder

    def use_first_of(self, name: str, primary: Node[T | None], fallback: Node[T]) -> Node[T]:
        """
        Ergonomic method to create a new node that returns the result of its
        ``input`` if it is provided and not ``None``, and otherwise returns the
        result of ``fallback``.  This method is used for things like filling in
        optional pipeline inputs.  For example, if you want the pipeline to take
        candidate items through an ``items`` input, but look them up from the
        user's history and the training data if ``items`` is not supplied, you
        would do:

        .. code:: python

            pipe = Pipeline()
            # allow candidate items to be optionally specified
            items = pipe.create_input('items', list[EntityId], None)
            # find candidates from the training data (optional)
            lookup_candidates = pipe.add_component(
                'select-candidates', UnratedTrainingItemsCandidateSelector(),
                user=history,
            )
            # if the client provided items as a pipeline input, use those; otherwise
            # use the candidate selector we just configured.
            candidates = pipe.use_first_of('candidates', items, lookup_candidates)

        .. note::

            This method does not distinguish between an input being unspecified
            and explicitly specified as ``None``.

        .. note::

            This method does *not* implement item-level fallbacks, only
            fallbacks at the level of entire results.  For item-level score
            fallbacks, see :class:`~lenskit.basic.FallbackScorer`.

        .. note::
            If one of the fallback elements is a component ``A`` that depends on
            another component or input ``B``, and ``B`` is missing or returns
            ``None`` such that ``A`` would usually fail, then ``A`` will be
            skipped and the fallback will move on to the next node. This works
            with arbitrarily-deep transitive chains.

        Args:
            name:
                The name of the node.
            primary:
                The node to use as the primary input, if it is available.
            fallback:
                The node to use if the primary input does not provide a value.
        """
        return self.add_component(name, fallback_on_none, primary=primary, fallback=fallback)

    def build(self) -> Pipeline:
        """
        Build the pipeline.
        """
        config = self.build_config()
        return Pipeline(config, [self._instantiate(n) for n in self._nodes.values()])

    def _instantiate(self, node: Node[ND]) -> Node[ND]:
        match node:
            case ComponentConstructorNode(name, constructor, config, connections=cxns):
                _log.debug("instantiating component", component=constructor)
                instance = constructor(config)
                return ComponentInstanceNode(name, instance, cxns)
            case _:
                return node

    def _check_available_name(self, name: str) -> None:
        if name in self._nodes or name in self._aliases:
            raise ValueError(f"pipeline already has node {name}")

    def _check_member_node(self, node: Node[Any]) -> None:
        nw = self._nodes.get(node.name)
        if nw is not node:
            raise PipelineError(f"node {node} not in pipeline")
