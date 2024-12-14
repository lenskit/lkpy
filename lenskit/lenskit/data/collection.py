from __future__ import annotations

import warnings
from collections import namedtuple
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Generator,
    Generic,
    Iterator,
    Literal,
    Mapping,
    NamedTuple,
    TypeAlias,
    TypeVar,
    overload,
)

import pandas as pd
import pyarrow as pa
from more_itertools import chunked
from pyarrow.parquet import ParquetDataset, ParquetWriter

from lenskit.diagnostics import DataWarning

from .items import ItemList
from .schemas import column_name, normalize_columns
from .types import ID, Column

K = TypeVar("K", bound=tuple)
KeySchema: TypeAlias = type[K] | tuple[str, ...]
GenericKey: TypeAlias = tuple[ID, ...]


class UserIDKey(NamedTuple):
    """
    Key type for user IDs.  This is used for :class:`item list collections
    <ItemListCollection>` that are keyed by user ID, a common setup for
    recommendation runs and
    """

    user_id: ID


KEY_CACHE: dict[tuple[str, ...], type[tuple]] = {("user_id",): UserIDKey}


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
    _list_schema: dict[str, pa.DataType]

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
        self._list_schema = {}

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

    @classmethod
    def from_df(cls, df: pd.DataFrame, key: type[K] | Sequence[Column] | Column, *others: Column):
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
            fields = _key_fields(key)
            columns = fields + others
        else:
            if isinstance(key, Column):
                key = [key]
            columns = tuple(key) + others
            fields = [column_name(c) for c in key]
            key = _create_key_type(*fields)  # type: ignore

        df = normalize_columns(df, *columns)
        ilc = cls(key)  # type: ignore
        for k, gdf in df.groupby(list(fields)):
            ilc.add(ItemList.from_df(gdf), *k)

        return ilc

    def to_df(self) -> pd.DataFrame:
        """
        Convert this item list collection to a data frame.

        .. warning::

            If this item list collection has any keys with empty lists, those
            lists will be excluded from the output.
        """
        fields = self._key_class._fields  # type: ignore
        return (
            pd.concat({k: il.to_df(numbers=False) for (k, il) in self._lists}, names=fields)
            .reset_index(fields)
            .reset_index(drop=True)
        )

    def to_arrow(self, *, batch_size: int = 5000) -> pa.Table:
        """
        Convert this item list collection to an Arrow table.

        The resulting table has one row per item list, with the item list
        contents an ``items`` column of a structured list type.  This preserves
        empty item lists for higher-fidelity data storage.

        Args:
            batch_size:
                The Arrow record batch size.
        """
        return pa.Table.from_batches(self._iter_record_batches())

    def save_parquet(
        self,
        path: PathLike[str],
        *,
        layout: Literal["native", "flat"] = "native",
        batch_size: int = 5000,
        compression: Literal["zstd", "gzip", "snappy", "lz4"] | None = "zstd",
    ) -> None:
        """
        Save this item list collection to a Parquet file.  This supports two
        types of Parquet files: “native” collections store one row per list,
        with the item list contents in a repeated structure column named
        ``items``; this layout fully preserves the item list collection,
        including empty item lists.  The “flat” layout is easier to work with in
        software such as Pandas, but cannot store empty item lists.

        Args:
            layout:
                The table layout to use.
            batch_size:
                The Arrow record batch size.
            compression:
                The compression scheme to use.
        """
        if layout == "flat":
            self.to_df().to_parquet(path, compression=compression)
            return

        writer = None
        try:
            for batch in self._iter_record_batches(batch_size):
                if writer is None:
                    writer = ParquetWriter(Path(path), batch.schema, compression=compression)
                writer.write_batch(batch)
        finally:
            if writer is not None:
                writer.close()

    @classmethod
    def load_parquet(cls, path: PathLike[str] | list[PathLike[str]]) -> ItemListCollection:
        """
        Load this item list from a Parquet file using the native layout.

        .. note::

            To load item list collections in the flat layout, use Pandas and
            :meth:`from_df`.

        Args:
            path:
                Path to the Parquet file to load.
        """
        if isinstance(path, list):
            path = [Path(p) for p in path]
        else:
            path = Path(path)
        dataset = ParquetDataset(path)  # type: ignore
        table = dataset.read()
        keys = table.drop("items")
        lists = table.column("items")
        ilc = ItemListCollection(keys.schema.names)
        for i, key in enumerate(keys.to_pylist()):
            il_data = lists[i].values
            ilc.add(ItemList.from_arrow(il_data), **key)

        return ilc

    def _iter_record_batches(self, batch_size: int = 5000) -> Generator[pa.RecordBatch, None, None]:
        for batch in chunked(self._lists, batch_size):
            keys = pa.RecordBatch.from_pylist([_key_dict(k) for (k, _il) in batch])
            schema = pa.list_(pa.struct(self._list_schema))
            rb = keys.add_column(
                keys.num_columns,
                "items",
                pa.array(
                    [il.to_arrow(type="array", columns=self._list_schema) for (_k, il) in batch],
                    schema,
                ),
            )
            yield rb

    @property
    def key_fields(self) -> tuple[str]:
        "The names of the key fields."
        return _key_fields(self._key_class)

    @property
    def key_type(self) -> type[K]:
        """
        The type of collection keys.
        """
        return self._key_class

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

    def add_from(self, other: ItemListCollection, **fields: ID):
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
                cf = key._asdict() | fields
                key = self._key_class(**cf)
            self._add(key, list)

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

    def lookup_projected(self, key: tuple) -> ItemList | None:
        """
        Look up an item list using a *projected* key.  A projected key is a key
        that may have additional fields beyond those defined by this collection,
        that are ignored for the purposes of lookup.

        Args:
            key:
                The key.  Must be a named tuple (e.g. a key obtained from
                another item list collection).

        Returns:
            The item list with the specified key, projected to this collection's
            key fields, or ``None`` if no such list exists.
        """
        kp = project_key(key, self._key_class)
        return self.lookup(kp)

    def lists(self) -> Iterator[ItemList]:
        "Iterate over item lists without keys."
        return (il for (_k, il) in self._lists)

    def keys(self) -> Iterator[K]:
        "Iterate over keys."
        return (k for (k, _il) in self._lists)

    def __len__(self):
        return len(self._lists)

    def __iter__(self):
        return iter(self._lists)

    def __getitem__(self, key: int) -> tuple[K, ItemList]:
        return self._lists[key]


def _key_fields(kt: type[tuple]) -> tuple[str]:
    "extract the fields from a key type"
    return kt._fields  # type: ignore


def _key_dict(kt: tuple[ID, ...]) -> Mapping[str, Any]:
    return kt._asdict()  # type: ignore


@overload
def _create_key(kt: type[K], *values: ID) -> K: ...
@overload
def _create_key(kt: Sequence[str], *values: ID) -> GenericKey: ...
def _create_key(kt: type[K] | Sequence[str], *values: ID) -> tuple[Any, ...]:
    if isinstance(kt, type):
        return kt(*values)  # type: ignore
    else:
        kt = _create_key_type(*kt)  # type: ignore
        return kt(*values)  # type: ignore


def _create_key_type(*fields: str) -> type[GenericKey]:
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

    if isinstance(key, target):
        return key

    try:
        return target._make(getattr(key, f) for f in target._fields)  # type: ignore
    except AttributeError as e:
        raise TypeError(f"source key is missing field {e.name}")
