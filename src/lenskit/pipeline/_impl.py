# pyright: strict
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Mapping
from uuid import NAMESPACE_URL, uuid5

from numpy.random import BitGenerator, Generator, SeedSequence
from typing_extensions import Any, Literal, TypeAlias, TypeVar, overload

from lenskit.data import Dataset
from lenskit.diagnostics import PipelineError
from lenskit.logging import get_logger
from lenskit.training import Trainable, TrainingOptions

from . import config
from .config import PipelineConfig
from .nodes import (
    ComponentConstructorNode,
    ComponentInstanceNode,
    ComponentNode,
    InputNode,
    LiteralNode,
    Node,
)
from .state import PipelineState

if TYPE_CHECKING:
    from .builder import PipelineBuilder

_log = get_logger(__name__)

# common type var for quick use
T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

NAMESPACE_LITERAL_DATA = uuid5(NAMESPACE_URL, "https://ns.lenskit.org/literal-data/")
CloneMethod: TypeAlias = Literal["config", "pipeline-config"]


class Pipeline:
    """
    LensKit recommendation pipeline.  This is the core abstraction for using
    LensKit models and other components to produce recommendations in a useful
    way.  It allows you to wire together components in (mostly) abitrary graphs,
    train them on data, and serialize pipelines to disk for use elsewhere.

    Pipelines should not be directly instantiated; they must be built with a
    :class:`~lenskit.pipeline.PipelineBuilder` class, or loaded from a
    configuration with :meth:`from_config`. If you have a scoring model and just
    want to generate recommenations with a default setup and minimal
    configuration, see :func:`~lenskit.pipeline.topn_pipeline` or
    :class:`~lenskit.pipeline.RecPipelineBuilder`.

    Pipelines are also :class:`~lenskit.training.Trainable`, and train all
    trainable components.

    Stability:
        Caller
    """

    _config: config.PipelineConfig
    _nodes: dict[str, Node[Any]]
    _edges: dict[str, dict[str, str]]
    _aliases: dict[str, Node[Any]]
    _default: Node[Any] | None = None
    _hash: str | None = None

    def __init__(
        self,
        config: config.PipelineConfig,
        nodes: Iterable[Node[Any]],
    ):
        self._nodes = {}
        for node in nodes:
            if isinstance(node, ComponentConstructorNode):
                raise RuntimeError("pipeline is not fully instantiated")
            self._nodes[node.name] = node
        self._edges = {name: cc.inputs for (name, cc) in config.components.items()}

        self._config = config
        self._aliases = {}
        for a, t in config.aliases.items():
            self._aliases[a] = self.node(t)
        if config.default:
            self._default = self.node(config.default)

    @property
    def config(self) -> PipelineConfig:
        """
        Get the pipline configuration.

        .. important::

            Do not modify the configuration returned, or it will become
            out-of-sync with the pipeline and likely not behave correctly.
        """
        return self._config

    @property
    def name(self) -> str | None:
        """
        Get the pipeline name (if configured).
        """
        return self._config.meta.name

    @property
    def version(self) -> str | None:
        """
        Get the pipeline version (if configured).
        """
        return self._config.meta.version

    def meta(self) -> config.PipelineMeta:
        """
        Get the metadata (name, version, hash, etc.) for this pipeline without
        returning the whole config.
        """
        return self._config.meta

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
            raise KeyError(node)

    def node_input_connections(self, node: str | Node[Any]) -> Mapping[str, Node[Any]]:
        """
        Get the input wirings for a node.
        """
        node = self.node(node)
        edges = self._edges.get(node.name, {})
        return {name: self.node(src) for (name, src) in edges.items()}

    def clone(self) -> Pipeline:
        """
        Clone the pipeline, **without** its trained parameters.

        Returns:
            A new pipeline with the same components and wiring, but fresh
            instances created by round-tripping the configuration.
        """
        return self.from_config(self._config)

    @property
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

        In LensKit 2025.1, the configuration hash is computed by computing the
        JSON serialization of the pipeline configuration *without* a hash and
        returning the hex-encoded SHA256 hash of that configuration.
        """
        assert self._config.meta.hash, "pipeline configuration has no hash"
        return self._config.meta.hash

    @staticmethod
    def from_config(config: object) -> Pipeline:
        """
        Reconstruct a pipeline from a serialized configuration.

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
        from .builder import PipelineBuilder

        config = PipelineConfig.model_validate(config)
        builder = PipelineBuilder.from_config(config)
        return builder.build()

    def modify(self) -> PipelineBuilder:
        """
        Create a pipeline builder from this pipeline in order to modify it.

        Pipelines cannot be modified in-place, but this method sets up a new
        builder that will create a modified copy of the pipeline.  Unmodified
        component instances are reused as-is.

        .. note::

            Since default connections are applied in
            :meth:`~lenskit.pipeline.PipelineBuilder.build`, the modifying
            builder does not have default connections.
        """
        from .builder import PipelineBuilder

        builder = PipelineBuilder()

        for node in self.nodes():
            match node:
                case InputNode(name, types=types):
                    if types is None:
                        types = set[type]()
                    builder.create_input(name, *types)
                case LiteralNode(name, value):
                    builder.literal(value, name=name)
                case ComponentConstructorNode(name, comp, config):
                    cn = builder.add_component(name, comp, config)
                case ComponentInstanceNode(name, comp):
                    cn = builder.add_component(name, comp)
                case _:  # pragma: nocover
                    raise RuntimeError(f"invalid node {node}")

        for n, t in self._aliases.items():
            builder.alias(n, t.name)

        for node in self.nodes():
            match node:
                case ComponentNode(name):
                    wiring = self._edges.get(name, {})
                    cn = builder.node(name)
                    builder.connect(cn, **{wn: builder.node(wt) for (wn, wt) in wiring.items()})
                case _:
                    pass

        if self.config.default:
            builder.default_component(self.config.default)

        return builder

    def train(self, data: Dataset, options: TrainingOptions | None = None) -> None:
        """
        Trains the pipeline's trainable components (those implementing the
        :class:`TrainableComponent` interface) on some training data.

        .. admonition:: Random Number Generation
            :class: note

            If :attr:`TrainingOptions.rng` is set and is not a generator or bit
            generator (i.e. it is a seed), then this method wraps the seed in a
            :class:`~numpy.random.SeedSequence` and calls
            :class:`~numpy.random.SeedSequence.spawn()` to generate a distinct
            seed for each component in the pipeline.

        Args:
            data:
                The dataset to train on.
            options:
                The training options.  If ``None``, default options are used.
        """
        log = _log.bind(pipeline=self.name)
        if options is None:
            options = TrainingOptions()

        if isinstance(options.rng, SeedSequence):
            seed = options.rng
        elif options.rng is None or isinstance(options.rng, (Generator, BitGenerator)):
            seed = None
        else:
            seed = SeedSequence(options.rng)

        log.info("training pipeline components")
        for node in self.nodes():
            match node:
                case ComponentInstanceNode(name, comp):
                    clog = log.bind(name=name, component=comp)
                    if isinstance(comp, Trainable):
                        # spawn new seed if needed
                        c_opts = options if seed is None else replace(options, rng=seed.spawn(1)[0])
                        clog.debug("training component")
                        comp.train(data, c_opts)
                    else:
                        clog.debug("training not required")
                case _:
                    pass

    @overload
    def run(self, /, **kwargs: object) -> object: ...
    @overload
    def run(self, node: str, /, **kwargs: object) -> object: ...
    @overload
    def run(self, nodes: tuple[str, ...], /, **kwargs: object) -> tuple[object, ...]: ...
    @overload
    def run(self, node: Node[T], /, **kwargs: object) -> T: ...
    @overload
    def run(self, nodes: tuple[Node[T1], Node[T2]], /, **kwargs: object) -> tuple[T1, T2]: ...
    @overload
    def run(
        self, nodes: tuple[Node[T1], Node[T2], Node[T3]], /, **kwargs: object
    ) -> tuple[T1, T2, T3]: ...
    @overload
    def run(
        self, nodes: tuple[Node[T1], Node[T2], Node[T3], Node[T4]], /, **kwargs: object
    ) -> tuple[T1, T2, T3, T4]: ...
    @overload
    def run(
        self,
        nodes: tuple[Node[T1], Node[T2], Node[T3], Node[T4], Node[T5]],
        /,
        **kwargs: object,
    ) -> tuple[T1, T2, T3, T4, T5]: ...
    def run(
        self,
        nodes: str | Node[Any] | tuple[str, ...] | tuple[Node[Any], ...] | None = None,
        /,
        **kwargs: object,
    ) -> object:
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
            The pipeline result.  If no nodes are supplied, this is the result
            of the default node.  If a single node is supplied, it is the result
            of that node. If a tuple of nodes is supplied, it is a tuple of
            their results.

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
        if nodes is None:
            if self._default:
                node_list = [self._default]
            else:
                raise RuntimeError("no node specified and pipeline has no default")
        elif isinstance(nodes, str) or isinstance(nodes, Node):
            node_list = [nodes]
        else:
            node_list = nodes

        state = self.run_all(*node_list, **kwargs)
        results = [state[self.node(n).name] for n in node_list]

        if node_list is nodes:
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
        2.  If no nodes are specified, it runs *all* nodes.  This has the
            consequence of running nodes that are not required to fulfill the
            last node (such scenarios typically result from using
            :meth:`use_first_of`).

        Args:
            nodes:
                The nodes to run, as positional arguments (if no nodes are
                specified, this method runs all nodes).
            kwargs:
                The inputs.

        Returns:
            The full pipeline state, with :attr:`~PipelineState.default` set to
            the last node specified.
        """
        from .runner import PipelineRunner

        runner = PipelineRunner(self, kwargs)
        node_list = [self.node(n) for n in nodes]
        _log.debug("running pipeline", name=self.name, nodes=[n.name for n in node_list])
        if not node_list:
            node_list = self.nodes()

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

    def _check_member_node(self, node: Node[Any]) -> None:
        nw = self._nodes.get(node.name)
        if nw is not node:
            raise PipelineError(f"node {node} not in pipeline")
