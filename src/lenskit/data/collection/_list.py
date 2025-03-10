import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Generic, overload

import pandas as pd
import pyarrow as pa

from lenskit.diagnostics import DataWarning

from ..adapt import Column, column_name, normalize_columns
from ..items import ItemList
from ._base import ItemListCollection, MutableItemListCollection
from ._keys import ID, GenericKey, K, Ko, create_key_type, key_dict, key_fields


class ListILC(MutableItemListCollection[K], Generic[K]):
    """
    Mutable item list collection backed by a Python list.
    """

    _key_class: type[K]
    _lists: list[tuple[K, ItemList]]
    _index: dict[K, int] | None = None
    _list_schema: dict[str, pa.DataType]

    def __init__(self, key: type[K] | Sequence[str], *, index: bool = True):
        """
        Create a new item list collection.
        """
        if isinstance(key, type):
            self._key_class = key
        else:
            self._key_class = create_key_type(*key)  # type: ignore

        self._lists = []
        if index:
            self._index = {}
        self._list_schema = {}

    @property
    def list_schema(self):
        return self._list_schema

    @overload
    @classmethod
    def from_dict(
        cls, data: Mapping[GenericKey | ID, ItemList], key: type[K]
    ) -> ItemListCollection[K]: ...
    @overload
    @classmethod
    def from_dict(
        cls, data: Mapping[GenericKey | ID, ItemList], key: Sequence[str] | str | None = None
    ) -> ItemListCollection[GenericKey]: ...
    @classmethod
    def from_dict(
        cls,
        data: Mapping[GenericKey | ID, ItemList],
        key: type[K] | Sequence[str] | str | None = None,
    ) -> ItemListCollection[Any]:
        """
        Create an item list collection from a dictionary.
        """
        if key is not None:
            if isinstance(key, str):
                key = (key,)
            ilc = ListILC(key)
        else:
            k = next(iter(data.keys()))
            if isinstance(k, tuple) and getattr(k, "_fields"):
                ilc = ListILC(type(k))
            else:
                warnings.warn(
                    "no key specified but data does not use named tuples, using default field 'id'",
                    DataWarning,
                )
                ilc = ListILC(["id"])

        for k, il in data.items():
            if isinstance(k, tuple):
                ilc.add(il, *k)
            else:
                ilc.add(il, k)

        return ilc

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        key: type[K] | Sequence[Column] | Column | None = None,
        *others: Column,
    ):
        """
        Create an item list collection from a data frame.

        .. note::

            Keys with empty item lists will be silently excluded from the output
            data.

        Args:
            df:
                The data frame to convert.
            key:
                The key type or field(s).  Can be specified as a single column
                name (or :class:`~lenskit.data.types.AliasedColumn`).
            others:
                Other columns to consider; primarily used to pass additional
                aliased columns to normalize other clumnes like the item ID.
        """
        if isinstance(key, type):
            fields = key_fields(key)
            columns = fields + others
        else:
            if key is None:
                warnings.warn(
                    "no key specified, inferring from _id columns", DataWarning, stacklevel=2
                )
                key = [n for n in df.columns if n.endswith("_id") and n != "item_id"]
            elif isinstance(key, Column):
                key = [key]
            columns = tuple(key) + others
            fields = [column_name(c) for c in key]
            key = create_key_type(*fields)  # type: ignore

        df = normalize_columns(df, *columns)
        ilc = cls(key)  # type: ignore
        for k, gdf in df.groupby(list(fields)):
            ilc.add(ItemList.from_df(gdf), *k)

        return ilc

    def add(self, list: ItemList, *fields: ID, **kwfields: ID):
        """
        Add a single item list to this list.

        Args:
            list:
                The item list to add.
            fields, kwfields:
                The key fields for this list.
        """
        key = self._key_class(*fields, **kwfields)  # type: ignore
        self._add(key, list)

    def add_from(self, other: ItemListCollection[Ko], **fields: ID):
        """
        Add all collection from another collection to this collection.  If field
        values are supplied, they are used to supplement or overwrite the keys
        in ``other``; a common use case is to add results from multiple
        recommendation runs and save them a single field.

        Args:
            other:
                The item list collection to incorporate into this one.
            fields:
                Additional key fields (must be specified by name).
        """
        for key, list in other:
            if fields:
                cf = key_dict(key) | fields
                key = self._key_class(**cf)
            self._add(key, list)  # type: ignore

    def _add(self, key: K, list: ItemList):
        self._lists.append((key, list))
        if self._index is not None:
            self._index[key] = len(self._lists) - 1
        for fn, ft in list.arrow_types().items():
            pft = self._list_schema.get(fn, None)
            if pft is None:
                self._list_schema[fn] = ft
            elif not ft.equals(pft):
                raise TypeError(f"incompatible item lists: field {fn} type {ft} != {pft}")

    @overload
    def lookup(self, key: tuple) -> ItemList | None: ...
    @overload
    def lookup(self, *key: ID, **kwkey: ID) -> ItemList | None: ...
    def lookup(self, *args, **kwargs) -> ItemList | None:
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

        try:
            return self._lists[self._index[key]][1]  # type: ignore
        except KeyError:
            return None

    def items(self) -> Iterator[tuple[K, ItemList]]:
        "Iterate over item lists and keys."
        return iter(self._lists)

    def __len__(self):
        return len(self._lists)

    def __getitem__(self, pos: int, /) -> tuple[K, ItemList]:
        return self._lists[pos]
