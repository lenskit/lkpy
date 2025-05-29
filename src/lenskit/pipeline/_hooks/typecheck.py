# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Input type checking hook.
"""

from typing import Any

from lenskit.diagnostics import PipelineError

from ..nodes import ComponentInstanceNode
from ..types import TypeExpr, is_compatible_data


def typecheck_input_data(
    node: ComponentInstanceNode[Any], input_name: str, input_type: TypeExpr | None, value: Any
) -> Any:
    """
    Hook to check that input data matches the component's declared input type.
    """
    if input_type and not is_compatible_data(value, input_type):
        if value is None:
            raise PipelineError(
                f"no data available for required input ❬{input_name}❭ on component ❬{node.name}❭"
            )
        raise TypeError(
            f"input ❬{input_name}❭ on component ❬{node.name}❭ has invalid type {type(value)} (expected {input_type})"  # noqa: E501
        )

    return value
