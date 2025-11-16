# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Relationship accessors for Dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sps
import torch
from numpy.typing import NDArray
from typing_extensions import Literal, overload, override

from lenskit._accel import data as _accel_data
from lenskit.diagnostics import FieldError
from lenskit.logging import get_logger
from lenskit.random import random_generator

from .items import ItemList
from .matrix import COOStructure, CSRStructure, SparseRowArray
from .schema import RelationshipSchema, id_col_name, num_col_name
from .types import ID, LAYOUT
from .vocab import Vocabulary

if TYPE_CHECKING:
    from .collection import ItemListCollection

_log = get_logger(__name__)


class RelationshipSet:
    """
    Representation for a set of relationship records.  This is the class for
    accessing general relationships, with arbitrarily many entity classes
    involved and repeated relationships allowed.

    For two-entity relationships without duplicates (including relationships
    formed by coalescing repeated relationships or interactions),
    :class:`MatrixRelationshipSet` extends this with additional capabilities.

    Relationship sets can be pickled or serialized, and will not save the entire
    dataset with them.  They are therefore safe to save as component elements
    during training processes.

    .. note::

        Client code does not need to construct this class; obtain instances from
        a dataset's :meth:`~lenskit.data.Dataset.relationships` or
        :meth:`~lenskit.data.Dataset.interactions` method.

    Stability:
        Caller
    """

    name: str
    """
    The name of the relationship class for these relationships.
    """
    schema: RelationshipSchema
    _matrix_set: dict[tuple[str, str], MatrixRelationshipSet]

    _table: pa.Table
    """
    The Arrow table of relationship information.
    """

    _vocabularies: dict[str, Vocabulary]
    _link_cols: list[str]

    def __init__(
        self,
        name: str,
        vocabularies: dict[str, Vocabulary],
        schema: RelationshipSchema,
        table: pa.Table,
    ):
        self.name = name
        self.schema = schema
        self._matrix_set = {}
        self._table = table

        self._vocabularies = vocabularies
        self._link_cols = [num_col_name(e) for e in schema.entities]

    def __getstate__(self):
        return {
            "name": self.name,
            "schema": self.schema,
            "columns": self._link_cols,
            "table": self._table,
            "vocabularies": self._vocabularies,
        }

    def __setstate__(self, state):
        self.name = state["name"]
        self.schema = state["schema"]
        self._link_cols = state["columns"]
        self._table = state["table"]
        self._vocabularies = state["vocabularies"]

    @property
    def is_interaction(self) -> bool:
        """
        Query whether these relationships represent interactions.
        """
        return self.schema.interaction

    @property
    def entities(self) -> list[str]:
        return [(c if e is None else e) for c, e in self.schema.entities.items()]

    @property
    def attribute_names(self) -> list[str]:
        return [c for c in self._table.column_names if c not in self._link_cols]

    def item_lists(self) -> ItemListCollection:
        """
        Get a view of this relationship set as an item list collection.

        Currently only implemented for :class:`MatrixRelationshipSet`, call
        :meth:`matrix` first.
        """
        raise NotImplementedError("item_lists only implemented for matrix relationship sets")

    def count(self):
        if "count" in self._table.column_names:
            count_column = self._table.column("count")
            count_column = count_column.fill_null(pa.scalar(1, type=pa.int8()))
            count = pc.sum(count_column)
            return count.as_py()

        return self._table.num_rows

    def co_occurrences(
        self, entity: str, *, group: str | list[str] | None = None, order: str | None = None
    ) -> sps.coo_array:
        """
        Count co-occurrences of the specified entity.  This is useful for
        counting item co-occurrences for association rules and probabilties, but
        also has other uses as well.

        This method supports both **ordered** and **unordered** co-occurrences.
        Unordered co-occurrences just count the number of times the two items
        appear together, and the resulting matrix is symmetric.

        For ordered co-occurrences, the interactions are ordered by the
        attribute specified by ``order``, and the resulting matrix ``M`` may not
        be symmetric.  ``M[i,j]`` counts the number of times item ``j`` has
        appeared **after** item ``i``.

        If ``group`` is specified, it controls the grouping for counting
        co-occurrences. For example, if a relationship connects the ``user``,
        ``session``, and ``item`` classes, then:

        - ``rs.co_occurrances("item")`` counts the number of times each pair of
          items appear together in a session.
        - ``rs.co_occurrances("item", group="user")`` counts the number of times
          each pair of items were interacted with by the same user, regardless
          of session.

        Args:
            entity:
                The name of the entity to count.
            group:
                The names of grouping entity classes for counting
                co-occurrences. The default is to use all entities that are not
                being counted.
            order:
                The name of an attribute to use for ordering interactions to
                compute sequential co-occurrences.

        Returns:
            A sparse matrix with the co-occurrence counts.
        """
        if isinstance(group, str):
            group = [group]
        elif group is None:
            group = [e for e in self.entities if e != entity]

        if entity in group:  # pragma: nocover
            raise ValueError("cannot group by and count the same entity")

        if len(group) > 1:
            raise NotImplementedError("multiple grouping entities not yet supported")

        # TODO: handle count columns

        gc = num_col_name(group[0])
        ec = num_col_name(entity)
        sorts = [(num_col_name(k), "ascending") for k in group]
        if order is not None:
            sorts.append((order, "ascending"))
        _log.debug("sorting table by %s", sorts)
        tbl = self._table.sort_by(sorts)
        _log.debug("counting co-occurrences")
        result = _accel_data.count_cooc(
            tbl.column(gc).combine_chunks(),
            tbl.column(ec).combine_chunks(),
            order is not None,
        )
        n = len(self._vocabularies[entity])

        indices = np.stack(
            [
                result.column("row").to_numpy(zero_copy_only=False),
                result.column("col").to_numpy(zero_copy_only=False),
            ],
            axis=0,
        )

        return sps.coo_array(
            (result.column("count").to_numpy(zero_copy_only=False), indices),
            shape=(n, n),
        )

    def arrow(self, *, attributes: str | list[str] | None = None, ids=False) -> pa.Table:
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
                    self._vocabularies[e].ids(table.column(num_col_name(e)).to_numpy())
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
        self, *, row_entity: str | None = None, col_entity: str | None = None
    ) -> MatrixRelationshipSet:
        """
        Convert this relationship set into a matrix, coalescing duplicate
        observations.

        .. versionchanged:: 2025.6

            Removed the fixed defaults for ``row_entity`` and ``col_entity``.

        Args:
            row_entity:
                The specified row entity of the matrix.  Defaults to the first
                entity in the relationship's list of involved entities.
            col_entity:
                The specified column entity of the matrix. Defaults to the last
                entity in the relationship's list of involved entities.
        """
        if row_entity is None:
            row_entity = self.entities[0]
        if col_entity is None:
            col_entity = self.entities[-1]

        mat = self._matrix_set.get((row_entity, col_entity), None)
        if mat is None:
            mat = self._make_matrix(row_entity=row_entity, col_entity=col_entity)
            self._matrix_set[(row_entity, col_entity)] = mat
        return mat

    def _make_matrix(
        self, *, row_entity: str = "user", col_entity: str = "item"
    ) -> MatrixRelationshipSet:
        if row_entity not in self.schema.entities.keys():
            raise FieldError(self.name, row_entity)

        if col_entity not in self.schema.entities.keys():
            raise FieldError(self.name, col_entity)

        if col_entity == row_entity:
            raise ValueError("row and column entity should not be the same")

        e_dict: dict[str, str | None]
        e_dict = {
            row_entity: self.schema.entities[row_entity],
            col_entity: self.schema.entities[col_entity],
        }
        matrix_schema = RelationshipSchema(
            entities=e_dict,
            interaction=self.schema.interaction,
            repeats=self.schema.repeats,
            attributes=self.schema.attributes,
        )
        new_table = self._table

        if "count" in new_table.column_names:
            null_filled = new_table["count"].fill_null(1)
            new_table = new_table.set_column(
                new_table.schema.get_field_index("count"),
                "count",
                null_filled,  # type: ignore
            )

        aggregates: list[tuple[str, str]]
        column_renames: dict[str, str]
        use_threads = True

        group_keys = [entity + "_num" for entity in e_dict]
        exclude_last = group_keys + ["timestamp", "count"]

        aggregates = [
            (col_name, "last")
            for col_name in new_table.column_names
            if col_name not in exclude_last
        ]
        column_renames = {
            f"{col_name}_last": col_name
            for col_name in new_table.column_names
            if col_name not in exclude_last
        }
        if aggregates:
            use_threads = False

        table_group = new_table.group_by(group_keys, use_threads=use_threads)

        if "timestamp" in new_table.column_names:
            aggregates.extend([("timestamp", "max"), ("timestamp", "min")])
            column_renames.update(
                {"timestamp_max": "timestamp", "timestamp_min": "first_timestamp"}
            )

        if "count" in new_table.column_names:
            aggregates.append(("count", "sum"))
            column_renames.update({"count_sum": "count"})
        else:
            aggregates.append((row_entity + "_num", "count"))
            column_renames.update({row_entity + "_num_count": "count"})

        aggregated_table = table_group.aggregate(aggregates)
        aggregated_table = aggregated_table.rename_columns(column_renames)
        aggregated_table = aggregated_table.sort_by([(k, "ascending") for k in group_keys])

        return MatrixRelationshipSet(self.name, self._vocabularies, matrix_schema, aggregated_table)


class MatrixRelationshipSet(RelationshipSet):
    """
    Two-entity relationships without duplicates, accessible in matrix form.

    .. note::

        Client code does not need to construct this class; obtain instances from
        a relationship set's :meth:`~RelationshipSet.matrix` method.
    """

    _row_ptrs: np.ndarray[tuple[int], np.dtype[np.int64]]
    _structure: SparseRowArray
    row_type: str
    _row_nums: pa.Int32Array
    _row_stats: pd.DataFrame | None = None

    col_type: str
    _col_nums: pa.Int32Array
    _col_stats: pd.DataFrame | None = None

    _coords: _accel_data.CoordinateTable

    def __init__(
        self,
        name: str,
        vocabularies: dict[str, Vocabulary],
        schema: RelationshipSchema,
        table: pa.Table,
    ):
        super().__init__(name, vocabularies, schema, table)
        self._init_structures()

    def _init_structures(self, *, ds_name: str | None = None):
        log = _log.bind(dataset=ds_name, relationship=self.name)

        # order the table to compute the sparse matrix
        log.debug("setting up entity information")
        entities = list(self.schema.entities.keys())
        row, col = entities

        self.row_type = row
        self.col_type = col

        e_cols = [num_col_name(e) for e in entities]

        self._table = self._table.combine_chunks()

        # set up the coordinate table
        # TODO: make this use the cached coordinate table if possible
        self._coords = _accel_data.CoordinateTable(2)
        self._coords.extend(self._table.select(e_cols).to_batches())

        # compute the row pointers
        log.debug("computing CSR data")
        n_rows = len(self.row_vocabulary)
        row_sizes = np.zeros(n_rows + 1, dtype=np.int64)
        self._row_nums = self._table.column(e_cols[0]).combine_chunks()  # type: ignore
        rsz_struct = pc.value_counts(self._row_nums)
        rsz_nums = rsz_struct.field("values")
        rsz_counts = rsz_struct.field("counts").cast(pa.int32())
        row_sizes[np.asarray(rsz_nums) + 1] = rsz_counts
        self._row_ptrs = np.cumsum(row_sizes, dtype=np.int64)  # type: ignore

        self._col_nums = self._table.column(e_cols[1]).combine_chunks()  # type: ignore
        self._structure = SparseRowArray.from_arrays(
            self._row_ptrs,
            self._col_nums,
            shape=(len(self.row_vocabulary), len(self.col_vocabulary)),
        )

        log.debug("relationship set ready to use")

    def __getstate__(self):
        return {
            "name": self.name,
            "schema": self.schema,
            "columns": self._link_cols,
            "table": self._table,
            "vocabularies": self._vocabularies,
        }

    def __setstate__(self, state):
        self.name = state["name"]
        self.schema = state["schema"]
        self._link_cols = state["columns"]
        self._table = state["table"]
        self._vocabularies = state["vocabularies"]
        self._init_structures()

    @property
    def row_vocabulary(self):
        "The vocabulary for row entities."
        return self._vocabularies[self.row_type]

    @property
    def col_vocabulary(self):
        "The vocabulary for column entities."
        return self._vocabularies[self.col_type]

    @property
    def n_rows(self):
        return len(self.row_vocabulary)

    @property
    def n_cols(self) -> int:
        return len(self.col_vocabulary)

    @override
    def matrix(
        self,
        *,
        row_entity: str | None = None,
        col_entity: str | None = None,
    ) -> MatrixRelationshipSet:
        if row_entity is not None and row_entity != self.entities[0]:  # pragma: nocover
            raise ValueError(f"row {row_entity} does not match matrix row {self.entities[0]}")
        if col_entity is not None and col_entity != self.entities[1]:  # pragma: nocover
            raise ValueError(f"column {col_entity} does not match matrix row {self.entities[1]}")

        # already a matrix relationship set
        return self

    @overload
    def csr_structure(self, *, format: Literal["numpy"] = "numpy") -> CSRStructure: ...
    @overload
    def csr_structure(self, *, format: Literal["arrow"]) -> SparseRowArray: ...
    def csr_structure(
        self, *, format: Literal["numpy", "arrow"] = "numpy"
    ) -> CSRStructure | SparseRowArray:
        """
        Get the compressed sparse row structure of this relationship matrix.
        """
        if format == "arrow":
            return self._structure
        else:
            n_rows = len(self.row_vocabulary)
            n_cols = len(self.col_vocabulary)

            colinds = self._table.column(num_col_name(self.col_type)).to_numpy()
            return CSRStructure(self._row_ptrs, colinds, (n_rows, n_cols))

    def coo_structure(self) -> COOStructure:
        """
        Get the compressed sparse row structure of this relationship matrix.
        """
        n_rows = len(self.row_vocabulary)
        n_cols = len(self.col_vocabulary)

        rowinds = self._table.column(num_col_name(self.row_type)).to_numpy()
        colinds = self._table.column(num_col_name(self.col_type)).to_numpy()
        return COOStructure(rowinds, colinds, (n_rows, n_cols))

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
    ) -> sps.coo_array[Any, tuple[int, int]]: ...
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
    ) -> sps.csr_array[Any, tuple[int, int]]: ...
    @overload
    def scipy(
        self, attribute: str | None = None, *, layout: LAYOUT = "csr", legacy: bool = False
    ) -> sps.sparray | sps.spmatrix: ...
    def scipy(
        self, attribute: str | None = None, *, layout: LAYOUT = "csr", legacy: bool = False
    ) -> sps.sparray | sps.spmatrix:
        """
        Get this relationship matrix as a SciPy sparse matrix.

        .. note::
            If the selected attribute has missing values, they are *omitted* from the
            returned matrix.

        Args:
            attribute:
                The attribute to return, or ``None`` to return an indicator-only
                sparse matrix (all observed values are 1).
            layout:
                The matrix layout to return.

        Returns:
            The sparse matrix.
        """
        n_rows = self.n_rows
        n_cols = self.n_cols
        nnz = self._table.num_rows

        colinds = self._table.column(num_col_name(self.col_type)).to_numpy()
        mask = None
        if attribute is None or (attribute == "count" and "count" not in self._table.column_names):
            values = np.ones(nnz, dtype=np.float32)
        else:
            value_col = self._table.column(attribute)
            if value_col.null_count:
                mask = value_col.is_valid()
                values = value_col.filter(mask).to_numpy()
                mask = mask.to_numpy()
            else:
                values = value_col.to_numpy()

        if layout == "csr" and mask is None:
            if legacy:
                return sps.csr_matrix((values, colinds, self._row_ptrs), shape=(n_rows, n_cols))
            else:
                return sps.csr_array((values, colinds, self._row_ptrs), shape=(n_rows, n_cols))
        else:
            rowinds = self._table.column(num_col_name(self.row_type)).to_numpy()
            if mask is not None:
                colinds = colinds[mask]
                rowinds = rowinds[mask]
            if legacy:
                mat = sps.coo_matrix((values, (rowinds, colinds)), shape=(n_rows, n_cols))
                if layout == "csr":
                    mat = mat.tocsr()
            else:
                mat = sps.coo_array((values, (rowinds, colinds)), shape=(n_rows, n_cols))
                if layout == "csr":
                    mat = mat.tocsr()
            return mat

    @overload
    def torch(
        self, attribute: str | None = None, *, layout: Literal["csr"] = "csr"
    ) -> torch.Tensor: ...
    @overload
    def torch(self, attribute: str | None = None, *, layout: Literal["coo"]) -> torch.Tensor: ...
    def torch(self, attribute: str | None = None, *, layout: LAYOUT = "csr") -> torch.Tensor:
        """
        Get this relationship matrix as a PyTorch sparse tensor.

        .. note::
            If the selected attribute has missing values, they are *omitted* from the
            returned matrix.

        Args:
            attribute:
                The attribute to return, or ``None`` to return an indicator-only
                sparse matrix (all observed values are 1).
            layout:
                The matrix layout to return.

        Returns:
            The sparse matrix.
        """
        n_rows = self.n_rows
        n_cols = self.n_cols
        nnz = self._table.num_rows

        colinds = self._table.column(num_col_name(self.col_type)).to_numpy()
        colinds = torch.tensor(np.require(colinds, requirements="W"))
        mask = None
        if attribute is None or (attribute == "count" and "count" not in self._table.column_names):
            values = torch.ones(nnz, dtype=torch.float32)
        else:
            mask = None
            value_col = self._table.column(attribute)
            if value_col.null_count:
                mask = value_col.is_valid()
                value_col = value_col.filter(mask)
                mask = mask.to_numpy()

            if pa.types.is_timestamp(self._table.field(attribute).type):
                value_col = value_col.cast(pa.timestamp("s")).cast(pa.int64())

            values = torch.tensor(value_col.to_numpy())

        if layout == "csr" and mask is None:
            return torch.sparse_csr_tensor(
                crow_indices=torch.tensor(self._row_ptrs),
                col_indices=colinds,
                values=values,
                size=(n_rows, n_cols),
            )
        else:
            rowinds = torch.tensor(self._table.column(num_col_name(self.row_type)).to_numpy())
            indices = torch.stack((rowinds, colinds))
            if mask is not None:
                indices = indices[:, torch.as_tensor(mask)]
            mat = torch.sparse_coo_tensor(
                indices=indices, values=values, size=(n_rows, n_cols)
            ).coalesce()
            if layout == "csr":
                mat = mat.to_sparse_csr()
            return mat

    def sample_negatives(
        self,
        rows: np.ndarray[tuple[int], np.dtype[np.int32]],
        *,
        weighting: Literal["uniform", "popular", "popularity"] = "uniform",
        n: int | None = None,
        verify: bool = True,
        max_attempts: int = 10,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.int32]:
        """
        Sample negative columns (columns with no observation recorded) for an
        array of rows. On a normal interaction matrix, this samples negative
        items for users.

        Args:
            rows:
                The row numbers.  Duplicates are allowed, and negative columns
                are sampled independently for each row. Must be a 1D array or
                tensor.
            weighting:
                The weighting for sampled negatives; ``uniform`` samples them
                uniformly at random, while ``popularity`` samples them
                proportional to their popularity (number of occurrences).
            n:
                The number of negatives to sample for each user.  If ``None``,
                a single-dimensional vector is returned.
            verify:
                Whether to verify that the negative items are actually negative.
                Unverified sampling is much faster but can return false
                negatives.
            max_attempts:
                When verification is on, the maximum attempts before giving up
                and returning a possible false negative.
            rng:
                A random number generator to use.
        """
        rng = random_generator(rng)

        _log.debug("samping negatives", nrows=len(rows), ncols=n)

        eff_n = n or 1

        if verify:
            rows = np.require(rows, np.int32)
            columns = _accel_data.sample_negatives(
                self._coords,
                rows,
                self.n_cols,
                n=eff_n,
                max_attempts=max_attempts,
                pop_weighted=weighting != "uniform",
                seed=rng.bit_generator.random_raw(),
            )

        elif weighting == "uniform":
            columns = self._sample_unweighted(rng, eff_n)
        elif weighting == "popular" or weighting == "popularity":
            columns = self._sample_weighted(rng, eff_n)
        else:  # pragma: nocover
            raise ValueError(f"unsupported weighting {weighting}")

        columns = np.require(columns, np.int32)
        if n is None:
            columns = columns.reshape(-1)
        else:
            columns = columns.reshape(-1, n)

        return columns

    def _sample_unweighted(self, rng: np.random.Generator, size: int | tuple[int, int]):
        return rng.integers(0, self.n_cols, size=size)

    def _sample_weighted(self, rng: np.random.Generator, size: int | tuple[int, int]):
        return rng.choice(self._col_nums.to_numpy(), size=size, replace=False)

    @overload
    def row_table(self, id: ID) -> pa.Table | None: ...
    @overload
    def row_table(self, *, number: int) -> pa.Table: ...
    def row_table(self, id: ID | None = None, *, number: int | None = None) -> pa.Table | None:
        """
        Get a single row of this interaction matrix as a table.
        """
        if number is None and id is None:  # pragma: noover
            raise ValueError("must provide one of id and number")

        if number is None:
            number = self.row_vocabulary.number(id, "none")
            if number is None:
                return None

        row_start = self._row_ptrs[number]
        row_end = self._row_ptrs[number + 1]

        tbl = self._table.slice(row_start, row_end - row_start)
        tbl = tbl.drop_columns(num_col_name(self.row_type))
        return tbl

    @overload
    def row_items(self, id: ID) -> ItemList | None: ...
    @overload
    def row_items(self, *, number: int) -> ItemList: ...
    def row_items(self, id: ID | None = None, *, number: int | None = None) -> ItemList | None:
        """
        Get a single row of this interaction matrix as an item list.  Only valid
        when the column entity class is ``item''.
        """
        if self.col_type != "item":
            raise RuntimeError("row_items() only valid for item-column matrices")

        tbl = self.row_table(id=id, number=number)  # type: ignore
        if tbl is None:
            return None

        return ItemList.from_arrow(tbl, vocabulary=self.col_vocabulary)

    def to_ilc(self) -> ItemListCollection:
        """
        Get the rows as an item list collection.

        .. deprecated:: 2025.6

            Deprecated alias for :meth:`item_lists`.
        """
        return self.item_lists()

    def item_lists(self) -> ItemListCollection:
        """
        Get the rows as an item list collection.
        """
        from .collection import ListILC

        if self.col_type != "item":
            raise RuntimeError("row_items() only valid for item-column matrices")

        # FIXME: make this a lot faster with Arrow
        ilc = ListILC([f"{self.row_type}_id"])
        for i in range(self.n_rows):
            ilc.add(self.row_items(number=i), self.row_vocabulary.id(i))
        return ilc

    def row_stats(self):
        if self._row_stats is None:
            self._row_stats = self._compute_stats(self.row_type, self.col_type, self.row_vocabulary)
        return self._row_stats

    def col_stats(self):
        if self._col_stats is None:
            self._col_stats = self._compute_stats(self.col_type, self.row_type, self.col_vocabulary)
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
        stats = stats.reindex(stat_vocab.index, fill_value=0)
        if "mean_rating" in stats.columns:
            stats.loc[stats["rating_count"] == 0, "mean_rating"] = np.nan
        if "first_time" in stats.columns:
            stats.loc[stats["count"] == 0, "first_time"] = pd.NaT
        if "last_time" in stats.columns:
            stats.loc[stats["count"] == 0, "last_time"] = pd.NaT

        return stats
