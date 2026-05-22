# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Types and functions for lazy values.  These are used mostly for pipeline inputs.
"""

# pyright: strict
from __future__ import annotations

from threading import Lock
from typing import Callable, Protocol

__all__ = ["Lazy", "lazy_value", "lazy_thunk"]


class Lazy[T](Protocol):
    """
    Type for lazily-computed values.

    This is frequently used in pipeline components.  If a pipeline component
    may or may not need one of its inputs, declare the type with this to only
    run it as needed:

    .. code:: python

        def my_component(input: str, backup: Lazy[str]) -> str:
            if input == 'invalid':
                return backup.get()
            else:
                return input

    Stability:
        Caller
    """

    def get(self) -> T:
        """
        Get the value behind this lazy instance.

        .. note::

            When used as a pipeline input this method invokes upstream
            components if they have not yet been run.  Therefore, it may fail if
            one of the required components fails or pipeline data checks fail.

        Raises:
            Exception:
                Exceptions raised by sources of the lazy data (e.g. a thunk, or
                upstream components) may be raised when this method is called.
            SkipComponent:
                Internal exception raised to indicate that no value is available
                and the calling component should be skipped.  Components
                generally do not need to handle this directly, as it is used to
                signal the pipeline runner.

                Only raised when used as a pipeline input.
        """
        ...


def lazy_value[T](value: T) -> Lazy[T]:
    """
    Create a lazy wrapper for an already-computed value.

    Args:
        value:
            The value to wrap in a lazy object.

    Returns:
        A lazy wrapper for the already-computed value.
    """
    return ConstLazy(value)


def lazy_thunk[T](thunk: Callable[[], T]) -> Lazy[T]:
    """
    Create a lazy value that calls the provided function to get the value as
    needed.

    Args:
        thunk:
            The function to call to supply a value.  Will only be called once.

    Returns:
        A :class:`Lazy` that will call the provided function.
    """
    return LazyThunk(thunk)


class ConstLazy[T]:
    _value: T

    def __init__(self, value: T):
        self._value = value

    def get(self) -> T:
        return self._value


class LazyThunk[T]:
    _thunk: Callable[[], T]
    _lock: Lock
    _cached: ConstLazy[T] | None = None

    def __init__(self, thunk: Callable[[], T]):
        self._thunk = thunk
        self._lock = Lock()

    def get(self):
        # fast-path: already computed
        cl = self._cached
        if cl is not None:
            return cl.get()

        with self._lock:
            if self._cached is None:
                val = self._thunk()
                self._cached = ConstLazy(val)

            return self._cached.get()
