# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Miscellaneous utility functions.
"""

import logging
from copy import deepcopy
from textwrap import dedent
from typing import Any, Protocol, TypeVar, runtime_checkable

from lenskit.algorithms import Algorithm

from .random import derivable_rng
from .timing import Stopwatch  # noqa: F401

try:
    import resource
except ImportError:
    resource = None

_log = logging.getLogger(__name__)

__all__ = [
    "Stopwatch",
    "derivable_rng",
    "clone",
    "clean_str",
]

A = TypeVar("A")


@runtime_checkable
class ParamContainer(Protocol):
    def get_params(self, deep=True) -> dict[str, Any]: ...


def clone(algo: A) -> A:
    """
    Clone an algorithm, but not its fitted data.  This is like
    :func:`sklearn.base.clone`, but may not work on arbitrary SciKit estimators.
    LensKit algorithms are compatible with SciKit clone, however, so feel free
    to use that if you need more general capabilities.

    This function is somewhat derived from the SciKit one.

    >>> from lenskit.algorithms.bias import Bias
    >>> orig = Bias()
    >>> copy = clone(orig)
    >>> copy is orig
    False
    >>> copy.damping == orig.damping
    True
    """
    _log.debug("cloning %s", algo)
    if isinstance(algo, Algorithm) or isinstance(algo, ParamContainer):
        params = algo.get_params(deep=False)

        sps = dict([(k, clone(v)) for (k, v) in params.items()])
        return algo.__class__(**sps)
    elif isinstance(algo, list):
        return [clone(a) for a in algo]  # type: ignore
    elif isinstance(algo, tuple):
        return tuple(clone(a) for a in algo)  # type: ignore
    else:
        return deepcopy(algo)


class LastMemo:
    def __init__(self, func, check_type="identity"):
        self.function = func
        self.check = check_type
        self.memory = None
        self.result = None

    def __call__(self, arg):
        if not self._arg_is_last(arg):
            self.result = self.function(arg)
            self.memory = arg

        return self.result

    def _arg_is_last(self, arg):
        if self.check == "identity":
            return arg is self.memory
        elif self.check == "equality":
            return arg == self.memory


def last_memo(func=None, check_type="identity"):
    if func is None:
        return lambda f: LastMemo(f, check_type)
    else:
        return LastMemo(func, check_type)


def cached(prop):
    """
    Decorator for property getters to cache the property value.
    """
    cache = "_cached_" + prop.__name__

    def getter(self):
        val = getattr(self, cache, None)
        if val is None:
            val = prop(self)
            setattr(self, cache, val)
        return val

    getter.__doc__ = prop.__doc__

    return property(getter)


def no_progress(obj, **kwargs):
    return obj


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
