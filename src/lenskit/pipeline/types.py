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
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,
    _GenericAlias,  # type: ignore # noqa: PLC2701
    get_args,
    get_origin,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAliasType, TypeForm

from lenskit.diagnostics import PipelineWarning

T = TypeVar("T", covariant=True)
"""
General type variable for generic container types or inputs.
"""

TypeExpr: TypeAlias = type | UnionType
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


class TypecheckWarning(UserWarning):
    "Warnings about type-checking logic."

    pass


class SkipComponent(Exception):
    "Internal exception used to skip an optional component."

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

    Stability:
        Caller
    """

    def get(self) -> T:
        """
        Get the value behind this lazy instance.

        .. note::

            As this method invokes upstream components if they have not yet been
            run, it may fail if one of the required components fails or pipeline
            data checks fail.

        Raises:
            Exception:
                Any exception raised by the component(s) needed to supply the
                lazy value may be raised when this method is called.
            SkipComponent:
                Internal exception raised to indicate that no value is available
                and the calling component should be skipped.  Components
                generally do not need to handle this directly, as it is used to
                signal the pipeline runner.
        """
        ...


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
    return any(_compat_type_inner(typ, target) for target in targets)


def _compat_type_inner(typ: type, target: TypeExpr) -> bool:
    """
    Check a type for compatibility with a single type.
    """
    # always compatible with Any
    if target == Any:
        return True

    # try a straight subclass check first, but gracefully handle incompatible types
    try:
        if issubclass(typ, target):
            return True
    except TypeError:
        # failing to check instance is fine, continue other checks
        pass

    # resolve type aliases
    if isinstance(target, TypeAliasType):
        return _compat_type_inner(typ, target.__value__)
    if isinstance(typ, TypeAliasType):
        return _compat_type_inner(typ.__value__, target)

    # expand union types
    if is_union_type(target):
        types = get_args(target)
        return is_compatible_type(typ, *types)

    # handle numeric hierarchy
    if typ == int and isinstance(target, type) and issubclass(target, (float, complex)):  # noqa: E721
        return True
    if typ == float and isinstance(target, type) and issubclass(target, complex):  # noqa: E721
        return True

    # try to handle generic types
    if isinstance(typ, (GenericAlias, _GenericAlias)):
        warnings.warn(f"cannot type-check generic type {typ}", TypecheckWarning)
        return _compat_type_inner(get_origin(typ), target)
    if isinstance(target, (GenericAlias, _GenericAlias)):
        return _compat_type_inner(typ, get_origin(target))

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
    return any(_compat_data_inner(obj, target) for target in targets)


def _compat_data_inner(obj: object, target: TypeExpr) -> bool:
    # always compatible with Any
    if target == Any:
        return True

    # try a straight subclass check first, but gracefully handle incompatible target types
    try:
        if isinstance(obj, target):  # type: ignore
            return True
    except TypeError:
        # failing to check instance is fine, continue other checks
        pass

    # resolve type aliases
    if isinstance(target, TypeAliasType):
        return _compat_data_inner(obj, target.__value__)

    # expand union types
    if is_union_type(target):
        types = get_args(target)
        return is_compatible_data(obj, *types)

    # resolve numeric type hierarchy
    if isinstance(obj, int) and isinstance(target, type) and issubclass(target, (float, complex)):  # noqa: E721
        return True
    if isinstance(obj, float) and isinstance(target, type) and issubclass(target, complex):  # noqa: E721
        return True

    if isinstance(target, TypeVar):
        # check type variable bounds (we can't fully resolve type variables)
        if target.__bound__ is None or is_compatible_data(obj, target.__bound__):
            return True

    # attempt to resolve generic types
    if isinstance(target, (GenericAlias, _GenericAlias)):
        origin = get_origin(target)

        if isinstance(obj, np.ndarray):
            if origin == np.ndarray:
                # check for type compatibility
                _sz, dtw = get_args(target)
                (dt,) = get_args(dtw)
            elif origin == NDArray:
                (dt,) = get_args(target)
            else:
                dt = None

            if dt is not None and issubclass(obj.dtype.type, dt):
                return True
        elif isinstance(origin, type) and isinstance(obj, origin):
            # this has holes, but is as close an approximation as we can get
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


def type_string(obj: type | FunctionType | None, *, final_sep: Literal[":", "."] = ":") -> str:
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
        if obj.__qualname__.endswith(f"<locals>.{obj.__name__}"):
            warnings.warn(
                f"component {obj.__qualname__} is local, configuration will not be usable",
                PipelineWarning,
                stacklevel=2,
            )
        else:  # pragma: nocover
            err = TypeError("nested objects not yet supported")
            err.add_note(f"name: {obj.__name__}")
            err.add_note(f"qualified name: {obj.__qualname__}")
            raise err

    mod_path = obj.__module__
    out_path = f"{mod_path}{final_sep}{obj.__qualname__}"
    short_mod_path = re.sub(r"\._.*", "", mod_path)
    short_path = f"{short_mod_path}{final_sep}{obj.__name__}"

    if short_mod_path != mod_path:
        short_mod = import_module(short_mod_path)
        if getattr(short_mod, obj.__name__, None) is obj:
            out_path = short_path
        else:  # pragma: nocover
            warnings.warn(f"{short_path} is not {out_path}, using long path", PipelineWarning)

    return out_path


def resolve_type_string(tstr: str) -> Any:
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
            if not parts:  # pragma: noccover
                raise ValueError(f"unparsable type string {tstr}")
            mod_name = ".".join(parts[:-1])
            typ_name = parts[-1]

        mod = import_module(mod_name)
        return getattr(mod, typ_name)


def is_union_type(ty: TypeForm[Any]):
    # TODO: update to 'isinstance' after Python 3.14 unification of types
    origin = get_origin(ty)
    return origin is Union or origin is UnionType
