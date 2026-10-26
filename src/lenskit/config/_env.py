# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os
import re


def env_flag(name: str, default: bool = False) -> bool:
    """
    Parse a flag value from the environment as a boolean.
    """
    value = os.environ.get(name)
    return parse_flag(value, default=default, name=name)


def parse_flag(
    value: str | bool | int | None, *, default: bool = False, name: str | None = None
) -> bool:
    """
    Parse a flag value as a boolean.

    Args:
        value: The value to parse.
        default: The default value to return if ``value`` is ``None``.

    Returns:
        The parsed boolean value.
    """

    if value is None:
        return default
    elif isinstance(value, bool):
        return value
    elif isinstance(value, int):
        return bool(value)
    elif re.match(r"^\d+$", value):
        return bool(int(value))
    elif re.match(r"^(?:t(?:rue)?|y(?:es)?)$", value, re.IGNORECASE):
        return True
    elif re.match(r"^(?:f(?:alse)?|n(?:o)?)$", value, re.IGNORECASE):
        return False
    elif name is not None:
        raise ValueError(f"unrecognized boolean value {name}={value}")
    else:
        raise ValueError(f"unrecognized boolean value {value}")
