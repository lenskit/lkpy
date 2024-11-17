from collections import namedtuple
from collections.abc import Sequence
from typing import Generic, NamedTuple, TypeAlias, TypeVar

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

    Item list collections are used to represent a variety of things, including
    test data and the results of a batch run.  They are, at their heart
    dictionary mapping collection keys (named tuples of identifiers) to item
    lists.

    The key schema can be specified either by a list of field names, or by
    providing a named tuple class (created by either :func:`namedtuple` or
    :class:`NamedTuple`) defining the key schema.

    This class exists, instead of using raw dictionaries, to consistently handle
    some of the nuances of multi-valued keys, and different collections having
    different key fields; for example, if a run produces item lists with both
    user IDs and sequence numbers, but your test data is only indexed by user
    ID, the *projected lookup* capabilities make it easy to find the test data
    to go with an item list in the run.
    """

    key_class: type[K]
    data: dict[K, ItemList]

    def __init__(self, key: type[K] | Sequence[str]):
        """
        Create a new item list collection.
        """
        if isinstance(key, type):
            self.key_class = key
        else:
            self.key_class = _create_key_type(*key)  # type: ignore


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
