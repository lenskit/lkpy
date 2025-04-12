# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections import namedtuple
from collections.abc import Sequence

from typing_extensions import (
    Any,
    NamedTuple,
    TypeAlias,
    TypeVar,
    overload,
)

from ..types import ID

GenericKey: TypeAlias = tuple[ID, ...]
K = TypeVar("K", bound=tuple, default=GenericKey)
"""
Fixed key type.
"""
Ko = TypeVar("Ko", bound=tuple, default=GenericKey)
"""
Fixed key type for an "other" collection.
"""
KL = TypeVar("KL", covariant=True, bound=tuple, default=GenericKey)
"""
Covariant key type for key lookup.
"""
KeySchema: TypeAlias = type[K] | tuple[str, ...]


class UserIDKey(NamedTuple):
    """
    Key type for user IDs.  This is used for :class:`item list collections
    <ItemListCollection>` that are keyed by user ID, a common setup for
    recommendation runs and
    """

    user_id: ID


KEY_CACHE: dict[tuple[str, ...], type[tuple]] = {("user_id",): UserIDKey}


def key_fields(kt: type[tuple]) -> tuple[str]:
    "extract the fields from a key type"
    return kt._fields  # type: ignore


def key_dict(kt: tuple[ID, ...]) -> dict[str, Any]:
    return kt._asdict()  # type: ignore


@overload
def create_key(kt: type[K], *values: ID) -> K: ...
@overload
def create_key(kt: Sequence[str], *values: ID) -> GenericKey: ...
def create_key(kt: type[K] | Sequence[str], *values: ID) -> tuple[Any, ...]:
    if isinstance(kt, type):
        return kt(*values)  # type: ignore
    else:
        kt = create_key_type(*kt)  # type: ignore
        return kt(*values)  # type: ignore


def create_key_type(*fields: str) -> type[GenericKey]:
    """
    Create a new key
    """
    assert isinstance(fields, tuple)
    kt = KEY_CACHE.get(fields, None)
    if kt is None:
        ktn = f"LKILCKeyType{len(KEY_CACHE) + 1}"
        kt = namedtuple(ktn, fields)
        # support pickling
        kt.__reduce__ = _reduce_generic_key  # type: ignore
        KEY_CACHE[fields] = kt
    return kt


def _reduce_generic_key(key):
    args = (key._fields,) + key
    return create_key, args


def project_key(key: tuple, target: type[K]) -> K:
    """
    Project a key onto a subset of its fields.  This is to enable keys to be
    looked up in other collections that are keyed on a subset of their fields,
    such as using a key consisting of a user ID and a sequence number to look up
    test data in a collection keyed only by user ID.
    """

    if isinstance(key, target):
        return key

    try:
        return target._make(getattr(key, f) for f in target._fields)  # type: ignore
    except AttributeError as e:
        raise TypeError(f"source key is missing field {e.name}")
