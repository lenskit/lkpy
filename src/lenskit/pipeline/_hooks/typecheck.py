# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Input type checking hook.
"""

from typing import Any

from ..components import ComponentInput
from ..nodes import ComponentInstanceNode
from ..types import SkipComponent, SkipInput, is_compatible_data


def typecheck_input_data(
    node: ComponentInstanceNode[Any],
    input: ComponentInput,
    value: Any,
    *,
    required: bool = True,
    **context,
) -> Any:
    """
    Hook to check that input data matches the component's declared input type.
    """
    if input.type and not is_compatible_data(value, input.type):
        if value is None:
            bad_type = None
        else:
            bad_type = type(value)
        msg = f"found {bad_type}, expected ❬{input.type}❭"
        if value is None and input.has_default:
            err = SkipInput(msg)
        elif value is None and not required:
            err = SkipComponent(msg)
        else:
            err = TypeError(msg)
        err.add_note(f"Error encountered on input ❬{input.name}❭ to component ❬{node.name}❭")
        raise err

    return value
