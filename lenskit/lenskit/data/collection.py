from __future__ import annotations

import warnings
from collections import namedtuple
from collections.abc import Sequence
from typing import Any, Generic, Mapping, NamedTuple, TypeAlias, TypeVar, overload

from lenskit.diagnostics import DataWarning

from .items import ItemList
from .types import ID

K = TypeVar("K", bound=tuple)
KeySchema: TypeAlias = type[K] | tuple[str, ...]
KEY_CACHE: dict[tuple[str, ...], type[tuple]] = {}


class UserIDKey(NamedTuple):
    """
    Key type for user IDs.  This is used for :class:`item list collections
    <ItemListCollection>` that are keyed by user ID, a common setup for
    recommendation runs and
    """

    user_id: ID


class GenericKey(tuple):
    """
    A generic key.
    """

    _fields: tuple[str, ...]

    def __init__(self, *values, fields: tuple[str, ...]):
        super().__init__(*values)
        self._fields = fields


class ItemListCollection(Generic[K]):
    """
    A collection of item lists.

    An item list collection consists of a sequence of item lists with associated
    *keys* following a fixed schema.  Item list collections support iteration
    (in order) and lookup by key. They are used to represent a variety of
    things, including test data and the results of a batch run.

    The key schema can be specified either by a list of field names, or by
    providing a named tuple class (created by either :func:`namedtuple` or
    :class:`NamedTuple`) defining the key schema.  Schemas should **not** be
    nested: field values must be scalars, not tuples or lists.  Keys should also
    be hashable.

    This class exists, instead of using raw dictionaries or lists, to
    consistently handle some of the nuances of multi-valued keys, and different
    collections having different key fields; for example, if a run produces item
    lists with both user IDs and sequence numbers, but your test data is only
    indexed by user ID, the *projected lookup* capabilities make it easy to find
    the test data to go with an item list in the run.

    Item list collections support lookup by index, like a list, returning a
    tuple of the key and list.  If they are constructed with ``index=True``,
    they also support lookup by _key_, supplied as either a tuple or an instance
    of the key type; in this case, the key is not returned.  If more than one
    item with the same key is inserted into the collection, then the _last_ one
    is returned (just like a dictionary).

    Args:
        key:
            The type (a NamedTuple class) or list of field names specifying the
            key schema.
        index:
            Whether or not to index lists by key to facilitate fast lookups.
    """

    _key_class: type[K]
    _lists: list[tuple[K, ItemList]]
    _index: dict[K, int] | None = None

    def __init__(self, key: type[K] | Sequence[str], *, index: bool = True):
        """
        Create a new item list collection.
        """
        if isinstance(key, type):
            self._key_class = key
        else:
            self._key_class = _create_key_type(*key)  # type: ignore

        self._lists = []
        if index:
            self._index = {}

    @overload
    @classmethod
    def from_dict(
        cls, data: Mapping[tuple[ID, ...] | ID, ItemList], key: type[K]
    ) -> ItemListCollection[K]: ...
    @overload
    @classmethod
    def from_dict(
        cls, data: Mapping[tuple[ID, ...] | ID, ItemList], key: Sequence[str] | str | None = None
    ) -> ItemListCollection[tuple[ID, ...]]: ...
    @classmethod
    def from_dict(
        cls,
        data: Mapping[tuple[ID, ...] | ID, ItemList],
        key: type[K] | Sequence[str] | str | None = None,
    ) -> ItemListCollection[Any]:
        """
        Create an item list collection from a dictionary.
        """
        if key is not None:
            if isinstance(key, str):
                key = (key,)
            ilc = ItemListCollection(key)
        else:
            k = next(iter(data.keys()))
            if isinstance(k, tuple) and getattr(k, "_fields"):
                ilc = ItemListCollection(type(k))
            else:
                warnings.warn(
                    "no key specified but data does not use named tuples, using default field 'id'",
                    DataWarning,
                )
                ilc = ItemListCollection(["id"])

        for k, il in data.items():
            if isinstance(k, tuple):
                ilc.add(il, *k)
            else:
                ilc.add(il, k)

        return ilc

    def add(self, list, *fields: ID):
        key = self._key_class(*fields)  # type: ignore
        self._lists.append((key, list))
        if self._index is not None:
            self._index[key] = len(self._lists) - 1

    @overload
    def lookup(self, key: tuple) -> ItemList: ...
    @overload
    def lookup(self, *key: ID, **kwkey: ID) -> ItemList: ...
    def lookup(self, *args, **kwargs) -> ItemList:
        """
        Look up a list by key.  If multiple lists have the same key, this
        returns the **last** (like a dictionary).

        This method can be called with the key tuple as a single argument (and
        this can be either the actual named tuple, or an ordinary tuple of the
        same length), or with the individual key fields as positional or named
        arguments.

        Args:
            key:
                The key tuple or key tuple fields.
        """
        if self._index is None:
            raise TypeError("cannot lookup on non-indexed collection")
        if len(args) != 1 or not isinstance(args[0], tuple):
            key = self._key_class(*args, **kwargs)
        else:
            key = args[0]

        return self._lists[self._index[key]][1]  # type: ignore

    def __len__(self):
        return len(self._lists)

    def __iter__(self):
        return iter(self._lists)

    def __getitem__(self, key: int) -> tuple[K, ItemList]:
        return self._lists[key]


def _create_key(kt: KeySchema[K], *values: ID) -> K:
    if isinstance(kt, type):
        return kt(*values)  # type: ignore
    else:
        kt = _create_key_type(*kt)  # type: ignore
        return kt(*values)  # type: ignore


def _create_key_type(*fields: str) -> type[tuple[ID, ...]]:
    """
    Create a new key
    """
    assert isinstance(fields, tuple)
    kt = KEY_CACHE.get(fields, None)
    if kt is None:
        ktn = f"LKILCKeyType{len(KEY_CACHE)+1}"
        kt = namedtuple(ktn, fields)
        # support pickling
        kt.__reduce__ = _reduce_generic_key  # type: ignore
        KEY_CACHE[fields] = kt
    return kt


def _reduce_generic_key(key):
    args = (key._fields,) + key
    return _create_key, args


def project_key(key: tuple, target: type[K]) -> K:
    """
    Project a key onto a subset of its fields.  This is to enable keys to be
    looked up in other collections that are keyed on a subset of their fields,
    such as using a key consisting of a user ID and a sequence number to look up
    test data in a collection keyed only by user ID.
    """

    try:
        return target._make(getattr(key, f) for f in target._fields)  # type: ignore
    except AttributeError as e:
        raise TypeError(f"source key is missing field {e.name}")
