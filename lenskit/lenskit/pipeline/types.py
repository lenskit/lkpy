# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: basic
from __future__ import annotations

import re
import warnings
from importlib import import_module
from types import GenericAlias, NoneType
from typing import (  # type: ignore
    Generic,
    Protocol,
    TypeVar,
    Union,
    _GenericAlias,
    get_args,
    get_origin,
)

import numpy as np

T = TypeVar("T", covariant=True)


class TypecheckWarning(UserWarning):
    "Warnings about type-checking logic."

    pass


class Lazy(Protocol, Generic[T]):
    """
    Type for accepting lazy inputs from the pipeline runner.  If your function
    may or may not need one of its inputs, declare the type with this to only
    run it as needed:

    .. code:: python

        def my_component(input: str, backup: Lazy[str]) -> str:
            if input == 'invalid':
                return backup.get()
            else:
                return input
    """

    def get(self) -> T:
        """
        Get the value behind this lazy instance.
        """
        ...


def is_compatible_type(typ: type, *targets: type) -> bool:
    """
    Make a best-effort check whether a type is compatible with at least one
    target type. This function is limited by limitations of the Python type
    system and the effort required to (re-)write a full type checker.  It is
    written to be over-accepting instead of over-restrictive, so it can be used
    to reject clearly incompatible types without rejecting combinations it
    cannot properly check.

    Args:
        typ:
            The type to check.
        targets:
            One or more target types to check against.

    Returns:
        ``False`` if it is clear that the specified type is incompatible with
        all of the targets, and ``True`` otherwise.
    """
    for target in targets:
        # try a straight subclass check first, but gracefully handle incompatible types
        try:
            if issubclass(typ, target):
                return True
        except TypeError:
            pass

        if isinstance(target, (GenericAlias, _GenericAlias)):
            tcls = get_origin(target)
            # if we're matching a raw type against a generic, just check the origin
            if isinstance(typ, GenericAlias):
                warnings.warn(f"cannot type-check generic type {typ}", TypecheckWarning)
                cls = get_origin(typ)
                if issubclass(cls, tcls):  # type: ignore
                    return True
            elif isinstance(typ, type):
                print(typ, type(typ))
                if issubclass(typ, tcls):  # type: ignore
                    return True
        elif typ == int and isinstance(target, type) and issubclass(target, (float, complex)):  # noqa: E721
            return True
        elif typ == float and issubclass(target, complex):  # noqa: E721
            return True

    return False


def is_compatible_data(obj: object, *targets: type) -> bool:
    """
    Make a best-effort check whether a type is compatible with at least one
    target type. This function is limited by limitations of the Python type
    system and the effort required to (re-)write a full type checker.  It is
    written to be over-accepting instead of over-restrictive, so it can be used
    to reject clearly incompatible types without rejecting combinations it
    cannot properly check.

    Args:
        typ:
            The type to check.
        targets:
            One or more target types to check against.

    Returns:
        ``False`` if it is clear that the specified type is incompatible with
        all of the targets, and ``True`` otherwise.
    """
    for target in targets:
        # try a straight subclass check first, but gracefully handle incompatible types
        try:
            if isinstance(obj, target):
                return True
        except TypeError:
            pass

        if get_origin(target) == Union:
            types = get_args(target)
            if is_compatible_data(obj, *types):
                return True
        elif isinstance(target, (GenericAlias, _GenericAlias)):
            tcls = get_origin(target)
            if isinstance(obj, np.ndarray) and tcls == np.ndarray:
                # check for type compatibility
                _sz, dtw = get_args(target)
                (dt,) = get_args(dtw)
                if issubclass(obj.dtype.type, dt):
                    return True
            elif isinstance(tcls, type) and isinstance(obj, tcls):
                warnings.warn(
                    f"cannot type-check object of type {type(obj)} against generic",
                    TypecheckWarning,
                )
                return True
        elif (
            isinstance(obj, int)
            and isinstance(target, type)
            and issubclass(target, (float, complex))
        ):  # noqa: E721
            return True
        elif isinstance(obj, float) and issubclass(target, complex):  # noqa: E721
            return True

    return False


def type_string(typ: type | None) -> str:
    """
    Compute a string representation of a type that is both resolvable and
    human-readable.  Type parameterizations are lost.
    """
    if typ is None or typ is NoneType:
        return "None"
    elif typ.__module__ == "builtins":
        return typ.__name__
    elif typ.__qualname__ == typ.__name__:
        return f"{typ.__module__}.{typ.__name__}"
    else:
        return f"{typ.__module__}:{typ.__qualname__}"


def parse_type_string(tstr: str) -> type:
    """
    Compute a string representation of a type that is both resolvable and
    human-readable.  Type parameterizations are lost.
    """
    if tstr == "None":
        return NoneType
    elif re.match(r"^\w+$", tstr):
        return __builtins__[tstr]
    else:
        if ":" in tstr:
            mod_name, typ_name = tstr.split(":", 1)
        else:
            # separate last element from module
            parts = re.match(r"(.*)\.(\w+)$", tstr)
            if not parts:
                raise ValueError(f"unparsable type string {tstr}")
            mod_name, typ_name = parts.groups()

        mod = import_module(mod_name)
        return getattr(mod, typ_name)
