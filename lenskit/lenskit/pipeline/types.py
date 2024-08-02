# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: basic
from __future__ import annotations

import warnings
from types import GenericAlias
from typing import Union, _GenericAlias, get_args, get_origin  # type: ignore

import numpy as np


class TypecheckWarning(UserWarning):
    "Warnings about type-checking logic."

    pass


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
        elif typ == int and issubclass(target, (float, complex)):  # noqa: E721
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
        elif isinstance(obj, int) and issubclass(target, (float, complex)):  # noqa: E721
            return True
        elif isinstance(obj, float) and issubclass(target, complex):  # noqa: E721
            return True

    return False
