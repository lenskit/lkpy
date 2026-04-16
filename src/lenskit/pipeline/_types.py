# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: basic
from __future__ import annotations

import re
import warnings
from importlib import import_module
from types import FunctionType, GenericAlias, NoneType, UnionType
from typing import (
    Any,
    TypeAliasType,
    TypeVar,
    Union,
    _GenericAlias,  # type: ignore # noqa: PLC2701
    get_args,
    get_origin,
)

import numpy as np
from typing_extensions import TypeForm

from lenskit.diagnostics import PipelineWarning, TypecheckWarning

type TypeExpr = type | UnionType | TypeAliasType
"""
Type for (resolved) type expressions.

This type is intended to encapsulate any fully-resolved type expression.

:class:`type` encapsulates many other types, including:

- :class:`~types.GenericAlias`
- :class:`~types.NoneType`
- :class:`~types.FunctionType`
- :class:`~types.MethodType`
- :class:`~typing.TypeVar`
"""


class SkipComponent(Exception):
    "Internal exception used to skip an optional component."

    pass


class SkipInput(Exception):
    "Internal exception used to skip an optional component input."

    pass


def is_compatible_type(typ: type, *targets: TypeExpr) -> bool:
    """
    Make a best-effort check whether a type is compatible with at least one
    target type. This function is limited by limitations of the Python type
    system and the effort required to (re-)write a full type checker.  It is
    written to be over-accepting instead of over-restrictive, so it can be used
    to reject clearly incompatible types without rejecting combinations it
    cannot properly check.

    Stability:
        Internal

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
        # resolve type aliases
        if isinstance(target, TypeAliasType):
            target = target.__value__
        if target == Any:
            return True

        # try a straight subclass check first, but gracefully handle incompatible types
        try:
            if issubclass(typ, target):
                return True
        except TypeError:
            # failing to check instance is fine, continue other checks
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
                if issubclass(typ, tcls):  # type: ignore
                    return True
        elif typ == int and isinstance(target, type) and issubclass(target, (float, complex)):  # noqa: E721
            return True
        elif typ == float and isinstance(target, type) and issubclass(target, complex):  # noqa: E721
            return True

    return False


def is_compatible_data(obj: object, *targets: TypeExpr) -> bool:
    """
    Make a best-effort check whether a type is compatible with at least one
    target type. This function is limited by limitations of the Python type
    system and the effort required to (re-)write a full type checker.  It is
    written to be over-accepting instead of over-restrictive, so it can be used
    to reject clearly incompatible types without rejecting combinations it
    cannot properly check.

    Stability:
        Internal

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
        # resolve type aliases
        if isinstance(target, TypeAliasType):
            target = target.__value__
        if target == Any:
            return True

        # try a straight subclass check first, but gracefully handle incompatible types
        try:
            if isinstance(obj, target):
                return True
        except TypeError:
            # failing to check instance is fine, continue other checks
            pass

        origin = get_origin(target)
        if origin == UnionType or origin == Union:
            types = get_args(target)
            if is_compatible_data(obj, *types):
                return True
        elif isinstance(target, TypeVar):
            # is this quite correct?
            if target.__bound__ is None or isinstance(obj, target.__bound__):
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
                # this has holes, but is as close an approximation as we can get
                return True
        elif (
            isinstance(obj, int)
            and isinstance(target, type)
            and issubclass(target, (float, complex))
        ):  # noqa: E721
            return True
        elif isinstance(obj, float) and isinstance(target, type) and issubclass(target, complex):  # noqa: E721
            return True

    return False


def is_instance_or_subclass(obj: Any, typ: type):
    """
    Query if an object is an instance or subclass of the specified type.
    """
    if isinstance(obj, type):
        return issubclass(obj, typ)
    else:
        return isinstance(obj, typ)


def make_importable_path(obj: type | FunctionType | None) -> str:
    """
    Compute a string representation of a class or function that is both
    resolvable and human-readable.  Type parameterizations are lost.  The
    resulting string can be imported with :func:`import_path_string`.

    Stability:
        Internal
    """

    if obj is None or obj is NoneType:
        return "None"
    elif obj.__module__ == "builtins":
        return obj.__name__

    if obj.__qualname__ != obj.__name__:
        raise TypeError("nested objects not yet supported")

    mod_path = obj.__module__
    out_path = f"{mod_path}.{obj.__qualname__}"
    short_mod_path = re.sub(r"\._.*", "", mod_path)
    short_path = f"{short_mod_path}.{obj.__name__}"

    if short_mod_path != mod_path:
        short_mod = import_module(short_mod_path)
        if getattr(short_mod, obj.__name__, None) is obj:
            out_path = short_path
        else:
            warnings.warn(f"{short_path} is not {out_path}, using long path", PipelineWarning)

    return out_path


def import_path_string(tstr: str) -> Any:
    """
    Resolve a type string into an actual type or function.  This parses a string
    referenceing a class or function (as returned by :func:`make_importable_path`),
    imports the module, and resolves the final member.

    Stability:
        Internal
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
            parts = tstr.split(".")
            if not parts:
                raise ValueError(f"unparsable type string {tstr}")
            mod_name = ".".join(parts[:-1])
            typ_name = parts[-1]

        mod = import_module(mod_name)
        return getattr(mod, typ_name)


def is_union_type(ty: TypeForm[Any]):
    # TODO: update to 'isinstance' after Python 3.14 unification of types
    return get_origin(ty) is Union or get_origin(ty) is UnionType
