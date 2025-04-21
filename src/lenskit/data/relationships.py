# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Relationship accessors for Dataset.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sps
import torch
from numpy.typing import NDArray
from typing_extensions import Literal, overload, override

from lenskit._accel import NegativeSampler, RowColumnSet
from lenskit.diagnostics import FieldError
from lenskit.logging import get_logger
from lenskit.random import random_generator

from .arrow import is_sorted
from .items import ItemList
from .matrix import COOStructure, CSRStructure, SparseRowArray
from .schema import RelationshipSchema, id_col_name, num_col_name
from .types import ID, LAYOUT, MAT_AGG
from .vocab import Vocabulary

if TYPE_CHECKING:
    from .dataset import Dataset

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

    _vocabularies: dict[str, Vocabulary]
    _link_cols: list[str]

    def __init__(
        self,
        ds: Dataset,
        name: str,
        schema: RelationshipSchema,
        table: pa.Table,
    ):
        self.name = name
        self.schema = schema
        self._table = table

        self._vocabularies = {e: ds.entities(e).vocabulary for e in schema.entities}
        self._link_cols = [num_col_name(e) for e in schema.entities]

    @property
    def is_interaction(self) -> bool:
        """
        Query whether these relationships represent interactions.
        """
        return self.schema.interaction

    @property
    def attribute_names(self) -> list[str]:
        return [c for c in self._table.column_names if c not in self._link_cols]

    def count(self):
        if "count" in self._table.column_names:  # pragma: nocover
            raise NotImplementedError()

        return self._table.num_rows

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
        self, *, combine: MAT_AGG | Mapping[str, MAT_AGG] | None = None
    ) -> MatrixRelationshipSet:  # pragma: nocover
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

    .. note::

        Client code does not need to construct this class; obtain instances from
        a relationship set's :meth:`~RelationshipSet.matrix` method.
    """

    _row_ptrs: np.ndarray[tuple[int], np.dtype[np.int32]]
    _structure: SparseRowArray
    row_vocabulary: Vocabulary
    row_type: str
    _row_nums: pa.Int32Array
    _row_stats: pd.DataFrame | None = None

    col_vocabulary: Vocabulary
    col_type: str
    _col_nums: pa.Int32Array
    _col_stats: pd.DataFrame | None = None

    _rc_set: RowColumnSet

    def __init__(
        self,
        ds: Dataset,
        name: str,
        schema: RelationshipSchema,
        table: pa.Table,
    ):
        super().__init__(ds, name, schema, table)
        self._init_structures(ds_name=ds.name)

    def _init_structures(self, *, ds_name: str | None = None, _trust_table_sort: bool = False):
        log = _log.bind(dataset=ds_name, relationship=self.name)

        # order the table to compute the sparse matrix
        log.debug("setting up entity information")
        entities = list(self.schema.entities.keys())
        row, col = entities

        self.row_type = row
        self.row_vocabulary = self._vocabularies[row]
        self.col_type = col
        self.col_vocabulary = self._vocabularies[col]

        e_cols = [num_col_name(e) for e in entities]
        log.debug("checking relationship table sorting")
        if _trust_table_sort or is_sorted(self._table, e_cols):
            log.debug("relationship table already sorted 😊")
        else:
            log.warning("sorting relationship table (might take time)")
            self._table = self._table.sort_by([(c, "ascending") for c in e_cols])

        self._table = self._table.combine_chunks()

        # compute the row pointers
        log.debug("computing CSR data")
        n_rows = len(self.row_vocabulary)
        row_sizes = np.zeros(n_rows + 1, dtype=np.int32())
        self._row_nums = self._table.column(e_cols[0]).combine_chunks()  # type: ignore
        rsz_struct = pc.value_counts(self._row_nums)
        rsz_nums = rsz_struct.field("values")
        rsz_counts = rsz_struct.field("counts").cast(pa.int32())
        row_sizes[np.asarray(rsz_nums) + 1] = rsz_counts
        self._row_ptrs = np.cumsum(row_sizes, dtype=np.int32)

        self._col_nums = self._table.column(e_cols[1]).combine_chunks()  # type: ignore
        self._structure = SparseRowArray.from_arrays(
            self._row_ptrs,
            self._col_nums,
            shape=(len(self.row_vocabulary), len(self.col_vocabulary)),
        )

        # make the index
        log.debug("computing row-column index")
        self._rc_set = RowColumnSet(self._structure)
        log.debug("relationship set ready to use")

    def __getstate__(self):
        return {f: v for (f, v) in self.__dict__.items() if f != "_rc_set"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._rc_set = RowColumnSet(self._structure)

    @property
    def n_rows(self):
        return len(self.row_vocabulary)

    @property
    def n_cols(self) -> int:
        return len(self.col_vocabulary)

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
        if attribute is None or (attribute == "count" and "count" not in self._table.column_names):
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
        n_rows = self.n_rows
        n_cols = self.n_cols
        nnz = self._table.num_rows

        colinds = self._table.column(num_col_name(self.col_type)).to_numpy()
        colinds = torch.tensor(np.require(colinds, requirements="W"))
        if attribute is None or (attribute == "count" and "count" not in self._table.column_names):
            values = torch.ones(nnz, dtype=torch.float32)
        elif pa.types.is_timestamp(self._table.field(attribute).type):
            vals = self._table.column(attribute)
            vals = vals.cast(pa.timestamp("s")).cast(pa.int64())
            values = torch.tensor(vals.to_numpy())
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
            sampler = NegativeSampler(self._rc_set, np.require(rows, np.int32), eff_n)  # type: ignore

            count = 0
            while nr := sampler.num_remaining():
                count += 1
                last = count >= max_attempts
                candidates = self._sample_columns(rng, nr, weighting)
                sampler.accumulate(np.require(candidates, np.int32), last)
                if last:
                    break

            columns = sampler.result()
        else:
            columns = self._sample_columns(rng, eff_n, weighting)

        columns = np.require(columns, np.int32)
        if n is None:
            columns = columns.reshape(-1)
        else:
            columns = columns.reshape(-1, n)

        return columns

    def _sample_columns(
        self,
        rng: np.random.Generator,
        size: int | tuple[int, int],
        weighting: Literal["uniform", "popular", "popularity"],
    ):
        match weighting:
            case "uniform":
                return self._sample_unweighted(rng, size)
            case "popular" | "popularity":
                return self._sample_weighted(rng, size)
            case _:
                raise ValueError(f"unknown weighting strategy {weighting}")

    def _sample_unweighted(self, rng: np.random.Generator, size: int | tuple[int, int]):
        return rng.choice(self.n_cols, size=size, replace=True)

    def _sample_weighted(self, rng: np.random.Generator, size: int | tuple[int, int]):
        return rng.choice(self._col_nums.to_numpy(), size=size, replace=True)

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

        return ItemList.from_arrow(tbl, vocabulary=self.col_vocabulary)

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
