# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Pipeline runner logic.
"""

# pyright: strict
import logging
from typing import Any, Literal, TypeAlias

from . import Pipeline, PipelineError
from .components import PipelineComponent
from .nodes import ComponentNode, FallbackNode, InputNode, LiteralNode, Node
from .types import is_compatible_data

_log = logging.getLogger(__name__)
State: TypeAlias = Literal["pending", "in-progress", "finished", "failed"]


class PipelineRunner:
    """
    Node status and results for a single pipeline run.

    This class operates recursively; pipelines should never be so deep that
    recursion fails.
    """

    pipe: Pipeline
    inputs: dict[str, Any]
    status: dict[str, State]
    state: dict[str, Any]

    def __init__(self, pipe: Pipeline, inputs: dict[str, Any]):
        self.pipe = pipe
        self.inputs = inputs
        self.status = {n.name: "pending" for n in pipe.nodes}
        self.state = {}

    def run(self, node: Node[Any], *, required: bool = True) -> Any:
        """
        Run the pipleline to obtain the results of a node.
        """
        status = self.status[node.name]
        if status == "finished":
            return self.state[node.name]
        elif status == "in-progress":
            raise PipelineError(f"pipeline cycle encountered at {node}")
        elif status == "failed":  # pragma: nocover
            raise RuntimeError(f"{node} previously failed")

        _log.debug("processing node %s", node)
        self.status[node.name] = "in-progress"
        try:
            self._run_node(node, required)
            self.status[node.name] = "finished"
        except Exception as e:
            _log.error("node %s failed with error %s", node, e)
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
            case ComponentNode(name, comp, inputs, wiring):
                self._run_component(name, comp, inputs, wiring, required)
            case FallbackNode(name, alts):
                self._run_fallback(name, alts)
            case _:  # pragma: nocover
                raise PipelineError(f"invalid node {node}")

    def _inject_input(self, name: str, types: set[type] | None, required: bool) -> None:
        val = self.inputs.get(name, None)
        if val is None and required and types and not is_compatible_data(None, *types):
            raise PipelineError(f"input {name} not specified")

        if val is not None and types and not is_compatible_data(val, *types):
            raise TypeError(f"invalid data for input {name} (expected {types}, got {type(val)})")

        self.state[name] = val

    def _run_component(
        self,
        name: str,
        comp: PipelineComponent[Any],
        inputs: dict[str, type | None],
        wiring: dict[str, str],
        required: bool,
    ) -> None:
        in_data = {}
        _log.debug("processing inputs for component %s", name)
        for iname, itype in inputs.items():
            src = wiring.get(iname, None)
            if src is not None:
                snode = self.pipe.node(src)
            else:
                snode = self.pipe.get_default(iname)

            if snode is None:
                ival = None
            else:
                if required and itype:
                    ireq = not is_compatible_data(None, itype)
                else:
                    ireq = False
                ival = self.run(snode, required=ireq)

            # bail out if we're trying to satisfy a non-required dependency
            if ival is None and itype and not is_compatible_data(None, itype) and not required:
                return None

            if itype and not is_compatible_data(ival, itype):
                if ival is None:
                    raise TypeError(
                        f"no data available for required input ❬{iname}❭ on component ❬{name}❭"
                    )
                raise TypeError(
                    f"input ❬{iname}❭ on component ❬{name}❭ has invalid type {type(ival)} (expected {itype})"  # noqa: E501
                )

            in_data[iname] = ival

        _log.debug("running component %s", name)
        self.state[name] = comp(**in_data)

    def _run_fallback(self, name: str, alternatives: list[Node[Any]]) -> None:
        for alt in alternatives:
            val = self.run(alt, required=False)
            if val is not None:
                self.state[name] = val
                return

        # got this far, no alternatives
        raise RuntimeError(f"no alternative for {name} returned data")
