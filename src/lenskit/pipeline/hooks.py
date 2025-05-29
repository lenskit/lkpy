# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Definition of pipeline hook protocols.
"""

from typing_extensions import Any, Protocol, TypeAlias

from .components import Component, PipelineFunction
from .nodes import ComponentInstanceNode
from .types import TypeExpr

ComponentObject: TypeAlias = Component | PipelineFunction


class ComponentInputHook(Protocol):
    """
    Inspect or process data as it passes to a component's input.

    As with all :ref:`hooks`, an input hook is a callable that is run at the
    appropriate stage of the input.
    """

    def __call__(
        self,
        node: ComponentInstanceNode[Any],
        input_name: str,
        input_type: TypeExpr | None,
        value: object,
    ) -> Any:
        """
        Inspect or process the component input data.

        Args:
            node:
                The component node being invoked.
            input_name:
                The name of the component's input that will receive the data.
            input_type:
                The type of data the component expects for this input, if one
                was specified in the component definition.
            value:
                The data value to be supplied.  This is declared
                :class:`object`, because its type is not known or guaranteed in
                the genral case.

        Returns:
            The value to pass to the component.  For inspection-only hooks, this
            will just be ``value``; hooks can also substitute alternative
            values depending on application needs.
        """
        pass
