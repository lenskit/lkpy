# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Pipeline runner logic.
"""

# pyright: strict
# pyright: reportPrivateUsage=false
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeAlias, TypeVar

import structlog

from lenskit.diagnostics import PipelineError
from lenskit.logging import get_logger, trace

from ._impl import Pipeline
from ._profiling import RunProfiler
from .components import ComponentInput, component_inputs
from .nodes import ComponentInstanceNode, InputNode, LiteralNode, Node
from .types import SkipComponent, SkipInput, is_compatible_data

_log = get_logger(__name__)
T = TypeVar("T")
State: TypeAlias = Literal["pending", "in-progress", "finished", "failed"]


class PipelineRunner:
    """
    Node status and results for a single pipeline run.

    This class operates recursively; pipelines should never be so deep that
    recursion fails.

    Stability:
        Internal
    """

    log: structlog.stdlib.BoundLogger
    pipe: Pipeline
    inputs: dict[str, Any]
    status: dict[str, State]
    state: dict[str, Any]
    _profiler: RunProfiler | None = None

    def __init__(
        self, pipe: Pipeline, inputs: dict[str, Any], *, profiler: RunProfiler | None = None
    ):
        self.log = _log.bind(pipeline=pipe.name)
        self.pipe = pipe
        self.inputs = inputs
        self.status = {n.name: "pending" for n in pipe.nodes()}
        self.state = {}
        self._profiler = profiler

    def run(self, node: Node[Any], *, required: bool = True) -> Any:
        """
        Run the pipleline to obtain the results of a node.
        """
        log = _log.bind(node=node.name)
        status = self.status[node.name]
        if status == "finished":
            return self.state[node.name]
        elif status == "in-progress":
            raise PipelineError(f"pipeline cycle encountered at {node}")
        elif status == "failed":  # pragma: nocover
            raise RuntimeError(f"{node} previously failed")

        trace(log, "processing node")
        self.status[node.name] = "in-progress"
        try:
            self._run_node(node, required)
            self.status[node.name] = "finished"
        except Exception as e:
            log.error("failed to run node", exc_info=e)
            self.status[node.name] = "failed"
            raise e

        try:
            return self.state[node.name]
        except KeyError as e:
            if required:
                raise PipelineError(f"no result for pipeline node {node.name}") from e
            else:
                return None

    def _run_node(self, node: Node[Any], required: bool) -> None:
        match node:
            case LiteralNode(name, value):
                self.state[name] = value
            case InputNode(name, types=types):
                self._inject_input(name, types, required)
            case ComponentInstanceNode():
                self._run_component(node, required)
            case _:  # pragma: nocover
                raise PipelineError(f"invalid node {node}")

    def _inject_input(self, name: str, types: set[type] | None, required: bool) -> None:
        val = self.inputs.get(name, None)
        if val is None and required and types and not is_compatible_data(None, *types):
            raise PipelineError(f"input {name} not specified")

        if val is not None and types and not is_compatible_data(val, *types):
            raise TypeError(f"invalid data for input {name} (expected {types}, got {type(val)})")

        trace(self.log, "injecting input", name=name, value=val)
        self.state[name] = val

    def _run_component(
        self,
        node: ComponentInstanceNode[Any],
        required: bool,
    ) -> None:
        in_data = {}
        log = self.log.bind(node=node.name)
        trace(log, "processing inputs")
        inputs = component_inputs(node.component, warn_on_missing=False)
        wiring = self.pipe.node_input_connections(node.name)
        for cin in inputs.values():
            itype = cin.type
            ilog = log.bind(input_name=cin.name, input_type=itype)
            trace(ilog, "resolving input")
            # look up the input wiring for this parameter input
            snode = None
            if src := wiring.get(cin.name, None):
                trace(ilog, "resolving from wiring")
                snode = self.pipe.node(src)

            ireq = required and not cin.accepts_none and not cin.has_default

            if cin.is_lazy:
                trace(ilog, "deferring input run")
                ival = DeferredRun(self, cin, node.name, snode, node, required=ireq)
            elif snode is None:
                trace(ilog, "input has no source node")
                ival = None
            else:
                trace(ilog, "running input node")
                ival = self.run(snode, required=ireq)

            if not cin.is_lazy:
                try:
                    ival = self._run_input_hooks(node, cin, ival, required=ireq)
                except SkipInput:
                    continue
                except SkipComponent:
                    return None

            in_data[cin.name] = ival

        trace(log, "running component", component=node.component)
        try:
            if self._profiler is not None:
                self._profiler.start_stage(node.name)
            self.state[node.name] = node.component(**in_data)
        except SkipComponent as e:
            trace(log, "component execution skipped", exc_info=e)
            self.state[node.name] = None
        finally:
            if self._profiler is not None:
                self._profiler.finish_stage(node.name)

    def _run_input_hooks(
        self,
        node: ComponentInstanceNode[Any],
        input: ComponentInput,
        value: Any,
        *,
        required: bool,
    ) -> Any:
        log = self.log.bind(node=node.name, input_name=input.name, input_type=input.type)
        for hook in self.pipe._run_hooks.get("component-input", []):
            trace(log, "running hook", hook=hook.function, required=required)
            try:
                value = hook.function(node, input, value, required=required)
            except SkipComponent as e:
                trace(log, "hook requested SkipComponent", hook=hook.function, exc_info=e)
                assert not required
                raise e
            except SkipInput as e:
                trace(log, "hook requested SkipInput", hook=hook.function, exc_info=e)
                assert input.accepts_none or input.has_default
                raise e
        return value


@dataclass(eq=False)
class DeferredRun(Generic[T]):
    """
    Implementation of :class:`Lazy` for deferred runs in a pipeline runner.

    Stability:
        Internal
    """

    runner: PipelineRunner
    input: ComponentInput
    cname: str
    node: Node[T] | None
    recv_node: ComponentInstanceNode[T]
    required: bool

    def get(self) -> T:
        if self.node is None:
            val = None
        else:
            val = self.runner.run(self.node, required=self.required)

        val = self.runner._run_input_hooks(self.recv_node, self.input, val, required=self.required)

        return val
