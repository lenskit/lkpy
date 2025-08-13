# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit dataset abstraction.
"""

# pyright: basic
from __future__ import annotations

import functools
import io
from abc import abstractmethod
from collections.abc import Callable
from os import PathLike

import pandas as pd
import pyarrow as pa
import scipy.sparse as sps
import torch
from humanize import metric
from numpy.typing import NDArray
from typing_extensions import Any, Literal, TypeAlias, TypedDict, TypeVar, overload

from lenskit.diagnostics import DataError
from lenskit.logging import get_logger

from .container import DataContainer
from .entities import EntitySet
from .items import ItemList
from .matrix import CSRStructure
from .relationships import MatrixRelationshipSet, RelationshipSet
from .schema import DataSchema, id_col_name
from .types import ID, LAYOUT
from .vocab import Vocabulary

_log = get_logger(__name__)

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


class DatasetState(TypedDict):
    data: DataContainer


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
        """
        Load a dataset in the LensKit native format.

        Args:
            path:
                The path to the dataset to load.

        Returns:
            The loaded dataset.
        """
        container = DataContainer.load(path)
        return cls(container)

    def save(self, path: str | PathLike[str]):
        """
        Save the data set in the LensKit native format.

        Args:
            path:
                The path in which to save the data set (will be created as a
                directory).
        """
        self._data.save(path)

    @property
    def name(self) -> str | None:
        """
        Get the dataset's name.
        """
        return self.schema.name

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
        log = _log.bind(dataset=self.name)
        self._data.normalize()

        self._entities = {}
        self._relationships = {}

        log.debug("initializing dataset caches")

        for name, schema in self._data.schema.entities.items():
            tbl = self._data.tables[name]
            id_name = id_col_name(name)
            ids = tbl.column(id_name)
            log.debug("creating entity vocabulary", entity=name)
            vocab = Vocabulary(ids, name=name, reorder=False)
            log.debug("creating entity set", entity=name)
            self._entities[name] = EntitySet(name, schema, vocab, tbl)

        for name, schema in self._data.schema.relationships.items():
            tbl = self._data.tables[name]
            relationship_vocab = {e: self.entities(e).vocabulary for e in schema.entities}
            if not schema.repeats.is_present and len(schema.entities) == 2:
                log.debug("creating matrix relationship set", relationship=name)
                self._relationships[name] = MatrixRelationshipSet(
                    name, relationship_vocab, schema, tbl
                )
            else:
                log.debug("creating relationship set", relationship=name)
                self._relationships[name] = RelationshipSet(name, relationship_vocab, schema, tbl)

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
                return iset.pandas(attributes=field, ids=original_ids)
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

    def __getstate__(self) -> DatasetState:
        self._ensure_loaded()
        return {
            "data": self._data,
        }

    def __setstate__(self, state: DatasetState):
        self._data = state["data"]
        self._init_caches()

    def __str__(self) -> str:
        s = "<Dataset"
        if self.name is not None:
            s += " " + self.name
        s += " ({} users, {} items, {} interactions)>".format(
            metric(self.user_count), metric(self.item_count), metric(self.interaction_count)
        )
        return s

    def __repr__(self) -> str:
        out = io.StringIO()
        out.write("<Dataset")
        if self.name is not None:
            out.write(" " + self.name)
        out.write(" {")
        for entity in self.schema.entities:
            eset = self._entities.get(entity, None)
            if eset is not None:
                out.write("  {}: {:,d},\n".format(entity, eset.count()))
            else:
                out.write("  {}: <not loaded>,\n".format(entity))
        for rel in self.schema.relationships:
            rset = self._relationships.get(rel, None)
            if rset is not None:
                out.write("  {}: {:,d},\n".format(rel, rset.count()))
            else:
                out.write("  {}: <not loaded>,\n".format(rel))
        out.write("}")

        return out.getvalue()
