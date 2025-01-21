# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit dataset abstraction.
"""

# pyright: basic
from __future__ import annotations

import functools
from abc import abstractmethod
from collections.abc import Callable, Mapping
from os import PathLike

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sps
import torch
from numpy.typing import NDArray
from typing_extensions import Any, Literal, TypeAlias, TypeVar, overload, override

from lenskit.diagnostics import DataError
from lenskit.logging import get_logger

from .container import DataContainer
from .items import ItemList
from .schema import DataSchema, EntitySchema, RelationshipSchema, id_col_name, num_col_name
from .types import ID, IDArray
from .vocab import Vocabulary

_log = get_logger(__name__)

DF_FORMAT: TypeAlias = Literal["numpy", "pandas", "torch"]
MAT_FORMAT: TypeAlias = Literal["scipy", "torch", "pandas", "structure"]
MAT_AGG: TypeAlias = Literal["count", "sum", "mean", "first", "last"]
LAYOUT: TypeAlias = Literal["csr", "coo"]
ACTION_FIELDS: TypeAlias = Literal["ratings", "timestamps"] | str

K = TypeVar("K")


def _uses_data(func):
    """
    Decorator to make sure the data is loaded.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self._ensure_loaded()
        return func(self, *args, **kwargs)

    return wrapper


class FieldError(KeyError):
    """
    The requested field does not exist.
    """

    def __init__(self, entity, field):
        super().__init__(f"{entity}[{field}]")


class Dataset:
    """
    Representation of a data set for LensKit training, evaluation, etc. Data can
    be accessed in a variety of formats depending on the needs of a component.
    See :ref:`data-model` for details of the LensKit data model.

    Dataset objects should not be directly constructed; instead, use a
    :class:`DatasetBuilder`, :meth:`load`, or :func:`from_interactions_df`.

    .. note::

        Zero-copy conversions are used whenever possible, so client code **must
        not** modify returned data in-place.

    .. todo::

        Support for advanced rating situations is not yet supported:

        * repeated ratings
        * mixed implicit & explicit feedback
        * later actions removing earlier ratings

    .. todo::

        Support for item and user content or metadata is not yet implemented.

    Args:
        data:
            The container for this dataset's data, or a function that will
            return such a container to create a lazy-loaded dataset.

    .. stability:: caller
    """

    _data: DataContainer
    _data_thunk: Callable[[], DataContainer | Dataset]
    _entities: dict[str, EntitySet]
    _relationships: dict[str, RelationshipSet]

    def __init__(self, data: DataContainer | Callable[[], DataContainer | Dataset]):
        if isinstance(data, DataContainer):
            self._data = data
            self._init_caches()
        else:
            self._data_thunk = data

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Dataset:
        container = DataContainer.load(path)
        return cls(container)

    def _ensure_loaded(self):
        if not hasattr(self, "_data"):
            _log.debug("lazy-loading dataset")
            data = self._data_thunk()
            del self._data_thunk
            if isinstance(data, DataContainer):
                self._data = data
                self._init_caches()
            elif isinstance(data, Dataset):
                data._ensure_loaded()
                self._data = data._data
                self._entities = data._entities
                self._relationships = data._relationships
            else:  # pragma: nocover
                raise TypeError("invalid thunk return: " + str(type(data)))

    def _init_caches(self):
        "Initialize internal caches for this dataset."

        self._entities = {}
        self._relationships = {}

        for name, schema in self._data.schema.entities.items():
            tbl = self._data.tables[name]
            id_name = id_col_name(name)
            ids = tbl.column(id_name)
            index = pd.Index(np.asarray(ids), name=id_name)
            vocab = Vocabulary(index, name=name)
            self._entities[name] = EntitySet(name, schema, vocab, tbl)

        for name, schema in self._data.schema.relationships.items():
            tbl = self._data.tables[name]
            if not schema.repeats.is_present and len(schema.entities) == 2:
                self._relationships[name] = MatrixRelationshipSet(self, name, schema, tbl)
            else:
                raise NotImplementedError("complex relationships not yet implemented")

    @property
    @_uses_data
    def schema(self) -> DataSchema:
        """
        Get the schema of this dataset.
        """
        return self._data.schema

    @property
    @_uses_data
    def items(self) -> Vocabulary:
        """
        The items known by this dataset.
        """
        return self.entities("item").vocabulary

    @property
    @_uses_data
    def users(self) -> Vocabulary:
        """
        The users known by this dataset.
        """
        return self.entities("user").vocabulary

    @property
    def item_count(self) -> int:
        return len(self.items)

    @property
    def user_count(self) -> int:
        return len(self.users)

    @_uses_data
    def entities(self, name: str) -> EntitySet:
        """
        Get the entities of a particular type / class.
        """
        eset = self._entities.get(name, None)
        if eset is None:
            raise DataError(f"entity class {name} is not defined")
        return eset

    @_uses_data
    def relationships(self, name: str) -> RelationshipSet:
        """
        Get the relationship records of a particular type / class.
        """
        rset = self._relationships.get(name, None)
        if rset is None:
            raise DataError(f"relationship class {name} is not defined")

        return rset

    @_uses_data
    def interactions(self, name: str | None = None) -> RelationshipSet:
        """
        Get the interaction records of a particular class.  If no class is
        specified, returns the default interaction class.
        """
        if name is None:
            name = self.default_interaction_class()
        rels = self.relationships(name)
        if not rels.is_interaction:
            raise DataError(f"relationship class {name} is not an interaction class")
        return rels

    def default_interaction_class(self) -> str:
        schema = self.schema
        if schema.default_interaction:
            return schema.default_interaction

        i_classes = [name for (name, rs) in schema.relationships.items() if rs.interaction]
        if len(i_classes) == 1:
            return i_classes[0]
        else:
            raise RuntimeError("no default interaction class specified")

    @property
    def interaction_count(self) -> int:
        """
        Count the total number of interactions of the default class, taking into
        account any ``count`` attribute.
        """
        return self.interactions().count()

    @overload
    @abstractmethod
    def interaction_table(
        self,
        *,
        format: Literal["pandas"],
        fields: str | list[str] | None = None,
        original_ids: bool = False,
    ) -> pd.DataFrame: ...
    @overload
    @abstractmethod
    def interaction_table(
        self, *, format: Literal["numpy"], fields: str | list[str] | None = None
    ) -> dict[str, NDArray[Any]]: ...
    @overload
    @abstractmethod
    def interaction_table(
        self, *, format: Literal["arrow"], fields: str | list[str] | None = None
    ) -> pa.Table: ...
    @abstractmethod
    @_uses_data
    def interaction_table(
        self,
        *,
        format: str,
        fields: str | list[str] | None = None,
        original_ids: bool = False,
    ) -> Any:
        """
        Get the user-item interactions as a table in the requested format. The
        table is not in a specified order.  Interactions may be repeated (e.g.
        the same user may listen to a song multiple times).  For a non-repeated
        “ratings matrix” view of the data, see :meth:`interaction_matrix`.

        This is a convenince wrapper on top of :meth:`interactions` and the
        methods of :class:`RelationshipSet`.

        .. warning::
            Client code **must not** perform in-place modifications on the table
            returned from this method.  Whenever possible, it will be a shallow
            view on top of the underlying storage, and modifications may corrupt
            data for other code.

        Args:
            format:
                The desired data format.  Currently-supported formats are:

                * ``"pandas"`` — returns a :class:`pandas.DataFrame`.  The index
                  is not meaningful.
                * ``"arrow"`` — returns a PyArrow :class:`~pa.Table`.  The index
                  is not meaningful.
                * ``"numpy"`` — returns a dictionary mapping names to arrays.
            fields:
                Which fields (attributes) to include, or ``None`` to include all
                fields. Commonly-available fields include ``"rating"`` and
                ``"timestamp"``.
            original_ids:
                If ``True``, return user and item IDs as represented in the
                original source data in columns named ``user_id`` and
                ``item_id``, instead of the user and item numbers typically
                returned.

        Returns:
            The user-item interaction log in the specified format.
        """
        iset = self.interactions()
        if format == "pandas":
            return iset.pandas(attributes=fields, ids=original_ids)
        else:
            table = iset.arrow(attributes=fields, ids=original_ids)
            if format == "numpy":
                return {c: table.column(c).to_numpy() for c in table.column_names}
            elif format == "arrow":
                return table
            else:
                raise ValueError(f"unsupported format {format}")

    @overload
    @abstractmethod
    def interaction_matrix(
        self,
        *,
        format: Literal["pandas"],
        field: str | None = None,
        original_ids: bool = False,
    ) -> pd.DataFrame: ...
    @overload
    @abstractmethod
    def interaction_matrix(
        self,
        *,
        format: Literal["torch"],
        layout: Literal["csr", "coo"] = "csr",
        field: str | None = None,
    ) -> torch.Tensor: ...
    @overload
    @abstractmethod
    def interaction_matrix(
        self,
        *,
        format: Literal["scipy"],
        layout: Literal["coo"],
        field: str | None = None,
    ) -> sps.coo_array: ...
    @overload
    @abstractmethod
    def interaction_matrix(
        self,
        *,
        format: Literal["scipy"],
        layout: Literal["csr"] = "csr",
        field: str | None = None,
    ) -> sps.csr_array: ...
    @overload
    @abstractmethod
    def interaction_matrix(
        self,
        *,
        format: Literal["structure"],
        layout: Literal["csr"] = "csr",
    ) -> CSRStructure: ...
    @abstractmethod
    @_uses_data
    def interaction_matrix(
        self,
        *,
        format: str,
        layout: LAYOUT = "csr",
        field: str | None = None,
        original_ids: bool = False,
        legacy: bool = False,
    ) -> Any:
        """
        Get the user-item interactions as “ratings” matrix from the default
        interaction class.  Interactions are not repeated, and are coalesced
        with the default coalescing strategy for each attribute.

        The matrix may be returned in “coordinate” format, in which case it is
        comparable to :meth:`interaction_table` but without repeated
        interactions, or it may be in a compressed sparse row format.

        This is a convenince wrapper on top of :meth:`interactions` and the
        methods of :class:`MatrixRelationshipSet`.

        .. warning::

            Client code **must not** perform in-place modifications on the
            matrix returned from this method.  Whenever possible, it will be a
            shallow view on top of the underlying storage, and modifications may
            corrupt data for other code.

        Args:
            format:
                The desired data format.  Currently-supported formats are:

                * ``"pandas"`` — returns a :class:`pandas.DataFrame`.
                * ``"torch"`` — returns a sparse :class:`torch.Tensor` (see
                  :mod:`torch.sparse`).
                * ``"scipy"`` — returns a sparse array from :mod:`scipy.sparse`.
                * ``"structure"`` — returns a :class:`~matrix.CSRStructure`
                  containing only the user and item numbers in compressed sparse
                  row format.
            field:
                Which field to return in the matrix.  Common fields include
                ``"rating"`` and ``"timestamp"``.

                If unspecified (``None``), this will yield an implicit-feedback
                indicator matrix, with 1s for observed items, except for the
                ``"pandas"`` format, which will return all attributes.  Specify
                an empty list to return a Pandas data frame with only the user
                and item attributes.
            layout:
                The layout for a sparse matrix.  Can be either ``csr`` or
                ``coo``, or ``None`` to use the default for the specified
                format.  Ignored for the Pandas format.
            original_ids:
                ``True`` to return user and item IDs instead of numbers in a
                ``pandas``-format matrix.
        """
        iset = self.interactions().matrix()

        match format:
            case "pandas":
                return iset.pandas(attributes=[field] if field else None, ids=original_ids)
            case "scipy":
                return iset.scipy(attribute=field, layout=layout, legacy=legacy)
            case "torch":
                return iset.torch(attribute=field, layout=layout)
            case "structure":
                if layout != "csr":
                    raise ValueError(f"unsupported layout {layout} for CSR structure")
                if field is not None:
                    raise ValueError("structure does not support fields")
                return iset.csr_structure()
            case _:
                raise ValueError(f"unknown matrix format {format}")

    @abstractmethod
    @overload
    def user_row(self, user_id: ID) -> ItemList | None: ...
    @abstractmethod
    @overload
    def user_row(self, *, user_num: int) -> ItemList: ...
    @abstractmethod
    def user_row(
        self, user_id: ID | None = None, *, user_num: int | None = None
    ) -> ItemList | None:
        """
        Get a user's row from the interaction matrix for the default interaction
        class, using :ref:`default coalsecing <coalescing-defaults>` for
        repeated interactions.  Available fields are returned as fields. If the
        dataset has ratings, these are provided as a ``rating`` field, **not**
        as the item scores.  The item list is unordered, but items are returned
        in order by item number.

        Args:
            user_id:
                The ID of the user to retrieve.
            user_num:
                The number of the user to retrieve.

        Returns:
            The user's interaction matrix row, or ``None`` if no user with that
            ID exists.
        """
        iset = self.interactions().matrix()
        return iset.row_items(id=user_id, number=user_num)

    def item_stats(self) -> pd.DataFrame:
        """
        Get item statistics from the default interaction class.

        Returns:
            A data frame indexed by item ID with the interaction statistics. See
            :ref:`interaction-stats` for a description of the columns returned.

            The index is the vocabulary, so ``iloc`` works with item numbers.
        """
        iset = self.interactions().matrix()
        if iset.col_type != "item":
            raise RuntimeError("default interactions do not have item columns")
        return iset.col_stats()

    def user_stats(self) -> pd.DataFrame:
        """
        Get user statistics from the default interaction class.

        Returns:
            A data frame indexed by user ID with the interaction statistics. See
            :ref:`interaction-stats` for a description of the columns returned.

            The index is the vocabulary, so ``iloc`` works with user numbers.
        """
        iset = self.interactions().matrix()
        if iset.row_type != "user":
            raise RuntimeError("default interactions do not have user columns")
        return iset.row_stats()


class EntitySet:
    """
    Representation of a set of entities from the dataset.  Obtained from
    :meth:`Dataset.entities`.
    """

    name: str
    """
    The name of the entity class for these entities.
    """
    schema: EntitySchema
    vocabulary: Vocabulary
    """
    The identifier vocabulary for this schema.
    """
    _table: pa.Table
    """
    The Arrow table of entity information.
    """

    def __init__(self, name: str, schema: EntitySchema, vocabulary: Vocabulary, table: pa.Table):
        self.name = name
        self.schema = schema
        self.vocabulary = vocabulary
        self._table = table

    def count(self) -> int:
        """
        Return the number of entities in this entity set.
        """
        return self._table.num_rows

    def ids(self) -> IDArray:
        """
        Get the identifiers of the entities in this set.  This is returned
        directly as PyArrow array instead of NumPy.
        """
        return self.vocabulary.ids()

    def numbers(self) -> np.ndarray[int, np.dtype[np.int32]]:
        """
        Get the numbers (from the vocabulary) for the entities in this set.
        """
        return np.arange(self.count(), dtype=np.int32)

    def arrow(self) -> pa.Table:
        """
        Get these entities and their attributes as a PyArrow table.
        """
        return self._table

    def pandas(self) -> pd.DataFrame:
        """
        Get the entities and their attributes as a Pandas data frame.
        """
        return self._table.to_pandas()

    def __len__(self):
        return self.count()


class RelationshipSet:
    """
    Representation for a set of relationship records.  This is the class for
    accessing general relationships, with arbitrarily many entity classes
    involved and repeated relationships allowed.

    For two-entity relationships without duplicates (including relationships
    formed by coalescing repeated relationships or interactions),
    :class:`MatrixRelationshipSet` extends this with additional capabilities.
    """

    dataset: Dataset
    """
    The dataset for these relationships.
    """

    name: str
    """
    The name of the relationship class for these relationships.
    """
    schema: RelationshipSchema

    _table: pa.Table
    """
    The Arrow table of relationship information.
    """

    _link_cols: list[str]

    def __init__(
        self,
        ds: Dataset,
        name: str,
        schema: RelationshipSchema,
        table: pa.Table,
    ):
        self.dataset = ds
        self.name = name
        self.schema = schema
        self._table = table
        self._link_cols = [num_col_name(e) for e in schema.entities]

    def is_interaction(self) -> bool:
        """
        Query whether these relationships represent interactions.
        """
        return self.schema.interaction

    def count(self):
        if "count" in self._table.column_names:
            raise NotImplementedError()

        return self._table.num_rows

    def arrow(self, *, attributes: str | list[str] | None, ids=False) -> pa.Table:
        """
        Get these relationships and their attributes as a PyArrow table.

        Args:
            attributes:
                The attributes to select.
            ids:
                If ``True``, include ID columns for the entities, instead of
                just the number columns.
        """
        table = self._table
        cols = self._link_cols

        if ids:
            id_cols = {}
            for e in self.schema.entity_class_names:
                id_cols[id_col_name(e)] = pa.array(
                    self.dataset.entities(e).vocabulary.ids(
                        table.column(num_col_name(e)).to_numpy()
                    )
                )
            id_tbl = pa.table(id_cols)
            cols = id_tbl.column_names
            for col in table.column_names:
                if col in self._link_cols:
                    continue

                id_tbl = id_tbl.append_column(col, table.column(col))
            table = id_tbl

        if attributes is not None:
            if isinstance(attributes, str):
                attr_cols = [attributes]
            else:
                attr_cols = attributes
            for ac in attr_cols:
                if ac not in table.column_names:
                    raise FieldError(self.name, ac)
            table = table.select(cols + attr_cols)

        return table

    def pandas(self, *, attributes: str | list[str] | None = None, ids=False) -> pd.DataFrame:
        """
        Get these relationship and their attributes as a PyArrow table.

        Args:
            attributes:
                The attributes to include in the resulting table.
            ids:
                If ``True``, include ID columns for the entities, instead of
                just the number columns.
        """
        tbl = self.arrow(attributes=attributes, ids=ids)
        return tbl.to_pandas()

    def matrix(
        self, *, combine: MAT_AGG | Mapping[str, MAT_AGG] | None = None
    ) -> MatrixRelationshipSet:
        """
        Convert this relationship set into a matrix, coalescing duplicate
        observations.

        Args:
            combine:
                The method for combining attribute values for repeated
                relationships or interactions.  Can either be a single strategy
                or a mapping from attribute names to combination strategies.
        """
        raise NotImplementedError()


class MatrixRelationshipSet(RelationshipSet):
    """
    Two-entity relationships without duplicates, accessible in matrix form.
    """

    _row_ptrs: np.ndarray[int, np.dtype[np.int32]]
    _row_vocab: Vocabulary
    row_type: str
    _row_stats: pd.DataFrame | None = None

    _col_vocab: Vocabulary
    col_type: str
    _col_stats: pd.DataFrame | None = None

    def __init__(
        self,
        ds: Dataset,
        name: str,
        schema: RelationshipSchema,
        table: pa.Table,
    ):
        super().__init__(ds, name, schema, table)
        # order the table to compute the sparse matrix
        entities = list(schema.entities.keys())
        row, col = entities
        self.row_type = row
        self._row_vocab = ds.entities(row).vocabulary
        self.col_type = col
        self._col_vocab = ds.entities(col).vocabulary

        e_cols = [num_col_name(e) for e in entities]
        table = table.sort_by([(c, "ascending") for c in e_cols])

        # compute the row pointers
        n_rows = len(self._row_vocab)
        row_sizes = np.zeros(n_rows + 1, dtype=np.int32())
        rsz_struct = pc.value_counts(table.column(e_cols[0]))
        rsz_nums = rsz_struct.field("values")
        rsz_counts = rsz_struct.field("counts").cast(pa.int32())
        row_sizes[np.asarray(rsz_nums) + 1] = rsz_counts
        self._row_ptrs = np.cumsum(row_sizes, dtype=np.int32)

    @override
    def matrix(
        self, *, combine: MAT_AGG | dict[str, MAT_AGG] | None = None
    ) -> MatrixRelationshipSet:
        # already a matrix relationship set
        return self

    def csr_structure(self) -> CSRStructure:
        """
        Get the compressed sparse row structure of this relationship matrix.
        """
        n_rows = len(self._row_vocab)
        n_cols = len(self._col_vocab)

        colinds = self._table.column(num_col_name(self.col_type)).to_numpy()
        return CSRStructure(self._row_ptrs, colinds, (n_rows, n_cols))

    @overload
    def scipy(
        self, attribute: str | None = None, *, layout: Literal["coo"], legacy: Literal[True]
    ) -> sps.coo_matrix: ...
    @overload
    def scipy(
        self,
        attribute: str | None = None,
        *,
        layout: Literal["coo"],
        legacy: Literal[False] = False,
    ) -> sps.coo_array: ...
    @overload
    def scipy(
        self, attribute: str | None = None, *, layout: Literal["csr"] = "csr", legacy: Literal[True]
    ) -> sps.csr_matrix: ...
    @overload
    def scipy(
        self,
        attribute: str | None = None,
        *,
        layout: Literal["csr"] = "csr",
        legacy: Literal[False] = False,
    ) -> sps.csr_array: ...
    @overload
    def scipy(
        self, attribute: str | None = None, *, layout: LAYOUT = "csr", legacy: bool = False
    ) -> sps.sparray | sps.spmatrix: ...
    def scipy(
        self, attribute: str | None = None, *, layout: LAYOUT = "csr", legacy: bool = False
    ) -> sps.sparray | sps.spmatrix:
        """
        Get this relationship matrix as a SciPy sparse matrix.

        Args:
            attribute:
                The attribute to return, or ``None`` to return an indicator-only
                sparse matrix (all observed values are 1).
            layout:
                The matrix layout to return.

        Returns:
            The sparse matrix.
        """
        n_rows = len(self._row_vocab)
        n_cols = len(self._col_vocab)
        nnz = self._table.num_rows

        colinds = self._table.column(num_col_name(self.col_type)).to_numpy()
        if attribute is None:
            values = np.ones(nnz, dtype=np.float32)
        else:
            values = self._table.column(attribute).to_numpy()

        if layout == "csr":
            if legacy:
                return sps.csr_matrix((values, colinds, self._row_ptrs), shape=(n_rows, n_cols))
            else:
                return sps.csr_array((values, colinds, self._row_ptrs), shape=(n_rows, n_cols))
        elif layout == "coo":
            rowinds = self._table.column(num_col_name(self.row_type))
            if legacy:
                return sps.coo_matrix((values, (rowinds, colinds)), shape=(n_rows, n_cols))
            else:
                return sps.coo_array((values, (rowinds, colinds)), shape=(n_rows, n_cols))

    @overload
    def torch(
        self, attribute: str | None = None, *, layout: Literal["csr"] = "csr"
    ) -> torch.Tensor: ...
    @overload
    def torch(self, attribute: str | None = None, *, layout: Literal["coo"]) -> torch.Tensor: ...
    def torch(self, attribute: str | None = None, *, layout: LAYOUT = "csr") -> torch.Tensor:
        """
        Get this relationship matrix as a PyTorch sparse tensor.

        Args:
            attribute:
                The attribute to return, or ``None`` to return an indicator-only
                sparse matrix (all observed values are 1).
            layout:
                The matrix layout to return.

        Returns:
            The sparse matrix.
        """
        n_rows = len(self._row_vocab)
        n_cols = len(self._col_vocab)
        nnz = self._table.num_rows

        colinds = self._table.column(num_col_name(self.col_type)).to_numpy()
        colinds = torch.tensor(colinds)
        if attribute is None:
            values = torch.ones(nnz, dtype=torch.float32)
        else:
            values = torch.tensor(self._table.column(attribute).to_numpy())

        if layout == "csr":
            return torch.sparse_csr_tensor(
                crow_indices=torch.tensor(self._row_ptrs),
                col_indices=colinds,
                values=values,
                size=(n_rows, n_cols),
            )
        elif layout == "coo":
            rowinds = torch.tensor(self._table.column(num_col_name(self.row_type)).to_numpy())
            indices = torch.stack((rowinds, colinds))
            return torch.sparse_coo_tensor(
                indices=indices, values=values, size=(n_rows, n_cols)
            ).coalesce()

    def row_table(self, id: ID | None = None, *, number: int | None = None) -> pa.Table | None:
        """
        Get a single row of this interaction matrix as a table.
        """
        if number is None and id is None:  # pragma: noover
            raise ValueError("must provide one of id and number")

        if number is None:
            number = self._row_vocab.number(id, "none")
            if number is None:
                return None

        row_start = self._row_ptrs[number]
        row_end = self._row_ptrs[number + 1]

        tbl = self._table.slice(row_start, row_end - row_start)
        tbl = tbl.drop_columns(num_col_name(self.row_type))
        return tbl

    def row_items(self, id: ID | None = None, *, number: int | None = None) -> ItemList | None:
        """
        Get a single row of this interaction matrix as an item list.  Only valid
        when the column entity class is ``item''.
        """
        if self.col_type != "item":
            raise RuntimeError("row_items() only valid for item-column matrices")

        tbl = self.row_table(id=id, number=number)
        if tbl is None:
            return None

        return ItemList.from_arrow(tbl, vocabulary=self._col_vocab)

    def row_stats(self):
        if self._row_stats is None:
            self._row_stats = self._compute_stats(self.row_type, self.col_type, self._row_vocab)
        return self._row_stats

    def col_stats(self):
        if self._col_stats is None:
            self._col_stats = self._compute_stats(self.col_type, self.row_type, self._col_vocab)
        return self._col_stats

    def _compute_stats(
        self, stat_type: str, other_type: str, stat_vocab: Vocabulary
    ) -> pd.DataFrame:
        s_col = num_col_name(stat_type)
        group = self._table.group_by(s_col)
        o_col = num_col_name(other_type)
        aggs = [(o_col, "count"), (o_col, "count_distinct")]
        if "count" in self._table.column_names:
            aggs.append(("count", "sum"))
        if "rating" in self._table.column_names:
            aggs += [("rating", "count"), ("rating", "mean")]
        if "first_time" in self._table.column_names:
            aggs.append(("first_time", "min"))
        if "last_time" in self._table.column_names:
            aggs.append(("last_time", "max"))
        if "timestamp" in self._table.column_names:
            aggs += [("timestamp", "min"), ("timestamp", "max")]

        stats = group.aggregate(aggs).to_pandas()
        stats = stats.rename(
            columns={
                o_col + "_count": "record_count",
                o_col + "_count_distinct": other_type + "_count",
                "count_sum": "count",
                "rating_mean": "mean_rating",
                "first_time_min": "first_time",
                "last_time_max": "last_time",
            }
        )
        if "count" not in stats.columns:
            stats["count"] = stats["record_count"]
        if "timestamp" in self._table.column_names:
            if "first_time" in self._table.columns:
                stats["first_time"] = np.minimum(stats["first_time"], stats["timestamp_min"])
            else:
                stats["first_time"] = stats["timestamp_min"]
            del stats["timestamp_min"]

            if "last_time" in self._table.columns:
                stats["last_time"] = np.maximum(stats["last_time"], stats["timestamp_max"])
            else:
                stats["last_time"] = stats["timestamp_max"]
            del stats["timestamp_max"]

        id_col = id_col_name(stat_type)
        stats[id_col] = stat_vocab.ids(stats[s_col])
        del stats[s_col]
        stats.set_index(id_col, inplace=True)
        stats = stats.reindex(stat_vocab.index)

        return stats


from .matrix import CSRStructure  # noqa: E402
