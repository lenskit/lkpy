# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"Data check functions for LensKit."

# pyright: strict
from __future__ import annotations

from typing import Any, Literal, Protocol, TypeVar, overload

import numpy as np
from numpy.typing import NDArray


class HasShape(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


A = TypeVar("A", bound=HasShape)
NPT = TypeVar("NPT", bound=np.generic)


@overload
def check_1d(
    arr: A,
    size: int | None = None,
    *,
    label: str = "array",
    error: Literal["raise"] = "raise",
) -> A: ...
@overload
def check_1d(
    arr: HasShape,
    size: int | None = None,
    *,
    error: Literal["return"],
) -> bool: ...
def check_1d(
    arr: A,
    size: int | None = None,
    *,
    label: str = "array",
    error: Literal["raise", "return"] = "raise",
) -> bool | A:
    """
    Check that an array is one-dimensional, optionally checking that it has the
    expected length.

    This check function has 2 modes:

    *   If ``error="raise"`` (the default), it will raise a :class:`TypeError`
        if the array shape is incorrect, and return the array otherwise.
    *   If ``error="return"``, it will return ``True`` or ``False`` depending on
        whether the size is correct.

    Stability:
        Caller

    Args:
        arr:
            The array to check.
        size:
            The expected size of the array. If unspecified, this function simply
            checks that the array is 1-dimensional, but does not check the size
            of that dimension.
        label:
            A label to use in the exception message.
        error:
            The behavior when an array fails the test.

    Returns:
        The array, if ``error="raise"`` and the array passes the check, or a
        boolean indicating whether it passes the check.

    Raises:
        TypeError: if ``error="raise"`` and the array fails the check.
    """
    if size is None and len(arr.shape) > 1:
        if error == "raise":
            raise TypeError(f"{label} must be 1D (has shape {arr.shape})")
        else:
            return False
    elif size is not None and arr.shape != (size,):
        if error == "raise":
            raise TypeError(f"{label} has incorrect shape (found {arr.shape}, expected {size})")
        else:
            return False

    if error == "raise":
        return arr
    else:
        return True


@overload
def check_type(
    arr: NDArray[Any],
    *types: type[NPT],
    label: str = "array",
    error: Literal["raise"] = "raise",
) -> NDArray[NPT]: ...
@overload
def check_type(
    arr: NDArray[Any],
    *types: type[NPT],
    error: Literal["return"],
) -> bool: ...
def check_type(
    arr: NDArray[Any],
    *types: type[NPT],
    label: str = "array",
    error: Literal["raise", "return"] = "raise",
) -> bool | NDArray[Any]:
    """
    Check that an array array is of an acceptable type.

    This check function has 2 modes:

    *   If ``error="raise"`` (the default), it will raise a :class:`TypeError`
        if the array shape is incorrect, and return the array otherwise.
    *   If ``error="return"``, it will return ``True`` or ``False`` depending on
        whether the size is correct.

    Stability:
        Caller

    Args:
        arr:
            The array to check.
        types:
            The acceptable types for the array.
        label:
            A label to use in the exception message.
        error:
            The behavior when an array fails the test.

    Returns:
        The array, if ``error="raise"`` and the array passes the check, or a
        boolean indicating whether it passes the check.

    Raises:
        TypeError: if ``error="raise"`` and the array fails the check.
    """
    if issubclass(arr.dtype.type, types):
        if error == "raise":
            return arr
        else:
            return True
    elif error == "raise":
        raise TypeError(f"{label} has incorrect type {arr.dtype} (allowed: {types})")
    else:
        return False
