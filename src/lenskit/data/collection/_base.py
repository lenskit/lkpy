from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Generator, Generic, Iterator, Literal, Mapping, Protocol, overload

import pandas as pd
import pyarrow as pa
from more_itertools import chunked
from pyarrow.parquet import ParquetDataset, ParquetWriter

from lenskit.diagnostics import DataWarning

from ..items import ItemList
from ..types import ID, Column
from ._keys import GenericKey, K, create_key_type, key_dict, key_fields, project_key


class ItemListCollection(Generic[K], ABC):
    """
    A collection of item lists.  This protocol defines read access to the
    collection; see :class:`ItemListCollector` for the ability to add new lists.

    An item list collection consists of a sequence of item lists with associated
    *keys* following a fixed schema.  Item list collections support iteration
    (in order) and lookup by key. They are used to represent a variety of
    things, including test data and the results of a batch run.

    The key schema can be specified either by a list of field names, or by
    providing a named tuple class (created by either :func:`namedtuple` or
    :class:`NamedTuple`) defining the key schema.  Schemas should **not** be
    nested: field values must be scalars, not tuples or lists.  Keys should also
    be hashable.

    This protocol and its implementations exist, instead of using raw
    dictionaries or lists, to consistently handle some of the nuances of
    multi-valued keys, and different collections having different key fields.
    For example, if a run produces item lists with both user IDs and sequence
    numbers, but your test data is only indexed by user ID, the *projected
    lookup* capabilities make it easy to find the test data to go with an item
    list in the run.

    Item list collections support indexing by position, like a list, returning a
    tuple of the key and list; iterating over an item list collection similarly
    produces ``(key, list)`` pairs (so an item list collection is a
    :class:`~collections.abc.Sequence` of key/list pairs).

    If the item list is _indexed_ (constructed with ``index=True``), it also
    supports lookup by _key_ with :meth:`lookup`.  The key can be supplied as
    either a tuple or an instance of the key type.  If more than one item with
    the same key is inserted into the collection, then the _last_ one is
    returned (just like a dictionary), but the others remain in the underlying
    list when it is iterated.

    .. note::

        Constructing an item list collection yields a
        :class:`~lenskit.data.ListILC`.

    Args:
        key:
            The type (a NamedTuple class) or list of field names specifying the
            key schema.
        index:
            Whether or not to index lists by key to facilitate fast lookups.
    """

    _key_class: type[K]

    def __new__(cls, key: type[K] | Sequence[str], *, index: bool = True):
        if cls == ItemListCollection or cls == MutableItemListCollection:
            return cls.empty(key, index=index)
        else:
            return super().__new__(cls)

    def __init__(self, key: type[K] | Sequence[str]):
        if isinstance(key, type):
            self._key_class = key
        else:
            self._key_class = create_key_type(*key)  # type: ignore

    @staticmethod
    def empty(key: type[K] | Sequence[str], *, index: bool = True) -> MutableItemListCollection[K]:
        """
        Create a new empty, mutable item list collection.
        """
        from ._list import ListILC

        return ListILC(key, index=index)

    @overload
    @staticmethod
    def from_dict(
        data: Mapping[GenericKey | ID, ItemList], key: type[K]
    ) -> ItemListCollection[K]: ...
    @overload
    @staticmethod
    def from_dict(
        data: Mapping[GenericKey | ID, ItemList], key: Sequence[str] | str | None = None
    ) -> ItemListCollection[GenericKey]: ...
    @staticmethod
    def from_dict(
        data: Mapping[GenericKey | ID, ItemList],
        key: type[K] | Sequence[str] | str | None = None,
    ) -> ItemListCollection[Any]:
        """
        Create an item list collection from a dictionary.

        .. seealso::
            :meth:`lenskit.data.collection.ListILC.from_dict`
        """
        from ._list import ListILC

        return ListILC.from_dict(data, key)

    @staticmethod
    def from_df(df: pd.DataFrame, key: type[K] | Sequence[Column] | Column, *others: Column):
        """
        Create an item list collection from a data frame.

        .. seealso::
            :meth:`lenskit.data.collection.ListILC.from_df`

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
        from ._list import ListILC

        return ListILC.from_df(df, key)

    def to_df(self) -> pd.DataFrame:
        """
        Convert this item list collection to a data frame.

        .. warning::

            If this item list collection has any keys with empty lists, those
            lists will be excluded from the output.
        """
        if any(len(il) == 0 for il in self.lists()):
            warnings.warn(
                "item list collection has empty lists, they will be dropped",
                DataWarning,
                stacklevel=2,
            )
        fields = list(self.key_fields)
        return (
            pd.concat({k: il.to_df(numbers=False) for (k, il) in self.items()}, names=fields)
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
        return pa.Table.from_batches(self.record_batches())

    def save_parquet(
        self,
        path: PathLike[str],
        *,
        layout: Literal["native", "flat"] = "native",
        batch_size: int = 5000,
        compression: Literal["zstd", "gzip", "snappy", "lz4"] | None = "zstd",
        mkdir: bool = True,
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
            mkdir:
                Whether to create the parent directories if they don't exist.
        """
        if mkdir:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        if layout == "flat":
            self.to_df().to_parquet(path, compression=compression)
            return

        writer = None
        try:
            for batch in self.record_batches(batch_size):
                if writer is None:
                    writer = ParquetWriter(Path(path), batch.schema, compression=compression)
                writer.write_batch(batch)
        finally:
            if writer is not None:
                writer.close()

    @overload
    @classmethod
    def load_parquet(
        cls,
        path: PathLike[str] | list[PathLike[str]],
        *,
        layout: Literal["native"] = "native",
    ) -> ItemListCollection: ...
    @overload
    @classmethod
    def load_parquet(
        cls,
        path: PathLike[str] | list[PathLike[str]],
        key: type[K] | Sequence[Column] | Column,
        *,
        layout: Literal["flat"],
    ) -> ItemListCollection: ...
    @classmethod
    def load_parquet(
        cls,
        path: PathLike[str] | list[PathLike[str]],
        key: type[K] | Sequence[Column] | Column | None = None,
        *,
        layout: Literal["native", "flat"] = "native",
    ) -> ItemListCollection:
        """
        Load this item list from a Parquet file.

        Args:
            path:
                Path to the Parquet file to load.
            key:
                The key to use (only when loading tabular layout).
            layout:
                The layout to use, either LensKit native layout or a flat tabular layout.
        """
        from ._list import ListILC

        if isinstance(path, list):
            path = [Path(p) for p in path]
        else:
            path = Path(path)
        dataset = ParquetDataset(path)  # type: ignore

        if layout == "native":
            if key is not None:
                raise ValueError("cannot specify key in native format")

            table = dataset.read()
            keys = table.drop("items")
            lists = table.column("items")
            ilc = ListILC(keys.schema.names)
            for i, key in enumerate(keys.to_pylist()):
                il_data = lists[i].values
                ilc.add(ItemList.from_arrow(il_data), **key)

            return ilc
        elif layout == "flat":
            tbl = dataset.read_pandas()

            if key is None:
                warnings.warn("no key specified, inferring from _id columns", DataWarning)
                key = [n for n in tbl.column_names if n.endswith("_id") and n != "item_id"]

            return cls.from_df(tbl.to_pandas(), key)
        else:  # pragma: nocover
            raise ValueError(f"unsupported layout {layout}")

    def record_batches(
        self, batch_size: int = 5000, columns: dict[str, pa.DataType] | None = None
    ) -> Generator[pa.RecordBatch, None, None]:
        """
        Get the item list collection as Arrow record batches (in native layout).
        """
        if columns is None:
            columns = self.list_schema

        for batch in chunked(self.items(), batch_size):
            keys = pa.Table.from_pylist([key_dict(k) for (k, _il) in batch])
            schema = pa.list_(pa.struct(columns))  # type: ignore
            tbl = keys.add_column(
                keys.num_columns,
                "items",
                pa.array(
                    [il.to_arrow(type="array", columns=columns) for (_k, il) in batch],
                    schema,
                ),
            )
            yield from tbl.to_batches()

    @property
    def key_fields(self) -> tuple[str]:
        "The names of the key fields."
        return key_fields(self.key_type)

    @property
    def key_type(self) -> type[K]:
        """
        The type of collection keys.
        """
        return self._key_class

    @property
    @abstractmethod
    def list_schema(self) -> dict[str, pa.DataType]:
        """
        Get the schema for the lists in this ILC.
        """

    @overload
    def lookup(self, key: tuple) -> ItemList | None: ...
    @overload
    def lookup(self, *key: ID, **kwkey: ID) -> ItemList | None: ...
    @abstractmethod
    def lookup(self, *args, **kwargs) -> ItemList | None:  # pragma: nocover
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
        raise NotImplementedError()

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

    @abstractmethod
    def items(self) -> Iterator[tuple[K, ItemList]]:
        "Iterate over item lists and keys."
        ...

    def lists(self) -> Iterator[ItemList]:
        "Iterate over item lists without keys."
        return (il for (_k, il) in self.items())

    def keys(self) -> Iterator[K]:
        "Iterate over keys."
        return (k for (k, _il) in self.items())

    @abstractmethod
    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[tuple[K, ItemList]]:
        return self.items()

    @abstractmethod
    def __getitem__(self, pos: int, /) -> tuple[K, ItemList]:
        """
        Get an item list and its key by position.

        Args:
            pos:
                The position in the list (starting from 0).

        Returns:
            The key and list at position ``pos``.

        Raises:
            IndexError:
                when ``pos`` is out-of-bounds.
        """
        pass


class ItemListCollector(Protocol):
    """
    Collect item lists with associated keys, as in :class:`ItemListCollection`.
    """

    @abstractmethod
    def add(self, list: ItemList, *fields: ID, **kwfields: ID):  # pragma: nocover
        """
        Add a single item list to this list.

        Args:
            list:
                The item list to add.
            fields, kwfields:
                The key fields for this list.
        """
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()


class MutableItemListCollection(ItemListCollector, ItemListCollection[K], Generic[K]):
    """
    Intersection type of :class:`ItemListCollection` and
    :class:`ItemListCollector`.
    """

    pass
