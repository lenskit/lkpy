# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Definition of pipeline hook protocols.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from types import FunctionType

from typing_extensions import (
    Any,
    NamedTuple,
    Protocol,
    TypedDict,
)

from lenskit.diagnostics import PipelineWarning

from ..components import Component, ComponentInput, PipelineFunction
from ..config import PipelineHook
from ..nodes import ComponentInstanceNode

type GenericHook = Callable[..., Any]
"""
Generic callable type for arbitrary hook functions.
"""
type ComponentObject = Component[Any] | PipelineFunction
"""
General type for instantiated component objects that may be passed to hooks.
"""


class HookEntry[Hook: GenericHook](NamedTuple):
    """
    An entry in a pipeline hook list.
    """

    function: Hook
    priority: int = 1

    def to_config(self) -> PipelineHook:
        if not isinstance(self.function, FunctionType):
            warnings.warn(f"hook {self.function} is not a function", PipelineWarning)
        function = f"{self.function.__module__}:{self.function.__qualname__}"  # type: ignore
        return PipelineHook(function=function, priority=self.priority)


class ComponentInputHook(Protocol):
    """
    Inspect or process data as it passes to a component's input.

    As with all :ref:`pipeline-hooks`, an input hook is a callable that is run
    at the appropriate stage of the input.

    Component input hooks are installed under the name ``component-input``.

    Stability:
        Experimental
    """

    def __call__(
        self,
        node: ComponentInstanceNode[Any],
        input: ComponentInput,
        value: object,
        **context: Any,
    ) -> Any:
        """
        Inspect or process the component input data.

        Args:
            node:
                The component node being invoked.
            input:
                The component input that will receive the data.
            value:
                The data value to be supplied.  This is declared
                :class:`object`, because its type is not known or guaranteed in
                the genral case.
            context:
                Additional context variables, mostly for the use of internal hooks.

        Returns:
            The value to pass to the component.  For inspection-only hooks, this
            will just be ``value``; hooks can also substitute alternative
            values depending on application needs.
        """
        pass


RunHooks = TypedDict(
    "RunHooks", {"component-input": list[HookEntry[ComponentInputHook]]}, total=False
)


def default_run_hooks() -> RunHooks:
    from .typecheck import typecheck_input_data

    return {"component-input": [HookEntry(typecheck_input_data, 0)]}
