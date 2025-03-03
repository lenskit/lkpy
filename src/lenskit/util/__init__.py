# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Miscellaneous utility functions.
"""

import logging
from textwrap import dedent
from typing import Any, Protocol, TypeVar, runtime_checkable

from ..random import derivable_rng, random_generator, set_global_rng

try:
    import resource
except ImportError:
    resource = None

_log = logging.getLogger(__name__)

__all__ = [
    "derivable_rng",
    "random_generator",
    "set_global_rng",
    "clean_str",
]

A = TypeVar("A")


@runtime_checkable
class ParamContainer(Protocol):
    def get_params(self, deep=True) -> dict[str, Any]: ...


def max_memory():
    "Get the maximum memory use for this process"
    if resource:
        res = resource.getrusage(resource.RUSAGE_SELF)
        return "%.1f MiB" % (res.ru_maxrss / 1024,)
    else:
        return "unknown"


def cur_memory():
    "Get the current memory use for this process"
    if resource:
        res = resource.getrusage(resource.RUSAGE_SELF)
        return "%.1f MiB" % (res.ru_idrss,)
    else:
        return "unknown"


def clean_str(s):
    return dedent(s).strip()
