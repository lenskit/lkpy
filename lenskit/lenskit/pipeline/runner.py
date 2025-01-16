# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Pipeline runner logic.
"""

# pyright: strict
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeAlias, TypeVar, get_args, get_origin

import structlog

from lenskit.diagnostics import PipelineError
from lenskit.logging import get_logger, trace

from ._impl import Pipeline
from .components import PipelineFunction, component_inputs
from .nodes import ComponentInstanceNode, InputNode, LiteralNode, Node
from .types import Lazy, is_compatible_data

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

    def __init__(self, pipe: Pipeline, inputs: dict[str, Any]):
        self.log = _log.bind(pipeline=pipe.name)
        self.pipe = pipe
        self.inputs = inputs
        self.status = {n.name: "pending" for n in pipe.nodes()}
        self.state = {}

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
                raise e
            else:
                return None

    def _run_node(self, node: Node[Any], required: bool) -> None:
        match node:
            case LiteralNode(name, value):
                self.state[name] = value
            case InputNode(name, types=types):
                self._inject_input(name, types, required)
            case ComponentInstanceNode(name, comp, wiring):
                self._run_component(name, comp, wiring, required)
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
        name: str,
        comp: PipelineFunction[Any],
        wiring: dict[str, str],
        required: bool,
    ) -> None:
        in_data = {}
        log = self.log.bind(node=name)
        trace(log, "processing inputs")
        inputs = component_inputs(comp)
        for iname, itype in inputs.items():
            # look up the input wiring for this parameter input
            snode = None
            if src := wiring.get(iname, None):
                snode = self.pipe.node(src)

            # check if this is a lazy node
            lazy = False
            if itype is not None and get_origin(itype) == Lazy:
                lazy = True
                (itype,) = get_args(itype)

            if snode is None:
                ival = None
            else:
                if required and itype:
                    ireq = not is_compatible_data(None, itype)
                else:
                    ireq = False

                if lazy:
                    ival = DeferredRun(self, iname, name, snode, required=ireq, data_type=itype)
                else:
                    ival = self.run(snode, required=ireq)

            # bail out if we're failing to satisfy a dependency but it is not required
            if (
                ival is None
                and itype
                and not lazy
                and not is_compatible_data(None, itype)
                and not required
            ):
                return None

            # check the data type before passing
            if itype and not lazy and not is_compatible_data(ival, itype):
                if ival is None:
                    raise TypeError(
                        f"no data available for required input ❬{iname}❭ on component ❬{name}❭"
                    )
                raise TypeError(
                    f"input ❬{iname}❭ on component ❬{name}❭ has invalid type {type(ival)} (expected {itype})"  # noqa: E501
                )

            in_data[iname] = ival

        trace(log, "running component", component=comp)
        self.state[name] = comp(**in_data)


@dataclass(eq=False)
class DeferredRun(Generic[T]):
    """
    Implementation of :class:`Lazy` for deferred runs in a pipeline runner.

    Stability:
        Internal
    """

    runner: PipelineRunner
    iname: str
    cname: str
    node: Node[T]
    required: bool
    data_type: type | None

    def get(self) -> T:
        val = self.runner.run(self.node, required=self.required)

        if self.data_type is not None and not is_compatible_data(val, self.data_type):
            raise TypeError(
                f"input ❬{self.iname}❭ on component ❬{self.cname}❭ has invalid type {type(val)} (expected {self.data_type})"  # noqa: E501
            )

        return val
