# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Data set builder.
"""

# pyright: basic
from __future__ import annotations

import datetime as dt
import json
import warnings
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import structlog
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_array, sparray

from lenskit._accel import data as _data_accel
from lenskit.data.matrix import SparseRowArray
from lenskit.data.vocab import Vocabulary
from lenskit.diagnostics import DataError, DataWarning
from lenskit.logging import get_logger

from .container import DataContainer
from .dataset import Dataset
from .schema import (
    AllowableTroolean,
    AttrLayout,
    ColumnSpec,
    DataSchema,
    EntitySchema,
    RelationshipSchema,
    check_name,
    id_col_name,
    num_col_name,
)
from .types import ID, NPID, CoreID, IDSequence  # noqa: F401

_log = get_logger(__name__)

NPT = TypeVar("NPT", bound=np.generic)
NPArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[NPT]]

TableInput: TypeAlias = pd.DataFrame | pa.Table | dict[str, NDArray[Any]]
RelationshipEntities: TypeAlias = Sequence[str] | Mapping[str, str | None]

DuplicateAction: TypeAlias = Literal["update", "error", "overwrite"]
"""
Action to take on duplicate entities.
"""
MissingEntityAction: TypeAlias = Literal["insert", "filter", "error"]
"""
Action to take when a relationship references a missing entity.
"""


class DatasetBuilder:
    """
    Construct data sets from data and tables.

    Args:
        name:
            The name of the new dataset, or a data container or dataset to
            use as the basis for building a new dataset.
    """

    schema: DataSchema
    """
    The data schema assembled so far.  Do not modify this schema directly.
    """

    _log: structlog.stdlib.BoundLogger
    _tables: dict[str, pa.Table | None]
    _vocabularies: dict[str, Vocabulary]
    _rel_coords: dict[str, _data_accel.CoordinateTable | None]

    def __init__(self, name: str | DataContainer | Dataset | None = None):
        self._rel_coords = {}

        if isinstance(name, Dataset):
            name._ensure_loaded()
            name = name._data
        if isinstance(name, DataContainer):
            self.schema = name.schema.model_copy()
            self._tables = {n: t for (n, t) in name.tables.items()}
            self._vocabularies = {
                n: Vocabulary(name.tables[n].column(id_col_name(n)), name=n)
                for n in name.schema.entities.keys()
            }
            # reuse _rel_coords if we have them
            if name._rel_coords is not None:
                self._rel_coords = {
                    n: coo.copy() for n, coo in name._rel_coords.items() if coo is not None
                }
        else:
            self.schema = DataSchema(name=name, entities={"item": EntitySchema()})
            self._tables = {"item": None}
            self._vocabularies = {}

        self._log = _log.bind(ds_name=name)

    @property
    def name(self) -> str | None:
        """
        Get the dataset name.
        """
        return self.schema.name

    def entity_classes(self) -> dict[str, EntitySchema]:
        """
        Get the entity classes defined so far.
        """
        return self.schema.entities

    def relationship_classes(self) -> dict[str, RelationshipSchema]:
        """
        Get the relationship classes defined so far.
        """
        return self.schema.relationships

    def record_count(self, class_name: str) -> int:
        """
        Get the number of records for the specified entity or relationship class.
        """
        tbl = self._tables[class_name]
        if tbl is None:
            return 0
        else:
            return tbl.num_rows

    def entity_id_type(self, name: str) -> pa.DataType:
        """
        Get the PyArrow data type for an entity classes's identifiers.
        """
        tbl = self._tables[name]
        if tbl is None:
            raise ValueError(f"entity class {name} has no entities")

        return tbl.field(id_col_name(name)).type

    def add_entity_class(self, name: str) -> None:
        """
        Add an entity class to the dataset.

        Args:
            name:
                The name of the entity class.
        """
        if name in self._tables:
            _log.debug(f"class name “{name}” already in use")
            return

        check_name(name)

        self._log.debug("adding entity class", class_name=name)
        self.schema.entities[name] = EntitySchema()
        self._tables[name] = None

    def add_relationship_class(
        self,
        name: str,
        entities: RelationshipEntities,
        allow_repeats: bool = True,
        interaction: bool = False,
    ) -> None:
        """
        Add a relationship class to the dataset.  This usually doesn't need to
        be called; :meth:`add_relationships` and :meth:`add_interactions` will
        automatically add the relationship class if needed.

        As noted in :ref:`data-model`, a *relationship* records a relationship
        or interaction between two or more *entities*.  Interactions are usually
        between users and items.  The ``entities`` option to this method defines
        the names of the entity classes participating.

        .. note::

            The order of entity classes in ``entities`` matters, as the
            relationship matrix logic
            (:meth:`lenskit.data.RelationshipSet.matrix`) will default to using
            the first and last entity classes as the rows and columns of the
            matrix.

        Args:
            name:
                The name of the relationship class.
            entities:
                The entity classes participating in the relationship class.
            allow_repeats:
                Whether repeated records for the same combination of entities
                are allowed.
            interaction:
                Whether this is an interaction relationship.
        """
        if name in self._tables:
            raise ValueError(f"relationship class name “{name}” already defined")

        check_name(name)

        self._log.debug("adding relationship class", class_name=name)
        e_dict: dict[str, str | None]
        if isinstance(entities, Mapping):
            e_dict = dict(entities.items())
        else:
            e_dict = {e: None for e in entities}
        enames = list(e_dict.keys())
        if interaction and enames[-1] != "item":
            warnings.warn(
                "the last entity class for relationship class {} is {}, exepected item".format(
                    name, enames[-1]
                ),
                DataWarning,
                stacklevel=2,
            )

        self.schema.relationships[name] = RelationshipSchema(
            entities=e_dict,
            interaction=interaction,
            repeats=AllowableTroolean.ALLOWED if allow_repeats else AllowableTroolean.FORBIDDEN,
        )
        self._tables[name] = _empty_rel_table(enames)

    @overload
    def add_entities(
        self,
        cls: str,
        ids: IDSequence | pa.Array[Any] | pa.ChunkedArray[Any],
        /,
        *,
        duplicates: DuplicateAction = "error",
    ) -> None: ...
    @overload
    def add_entities(
        self, cls: str, frame: TableInput, /, *, duplicates: DuplicateAction = "error"
    ) -> None: ...
    def add_entities(
        self,
        cls: str,
        source: IDSequence | pa.Array[Any] | pa.ChunkedArray[Any] | TableInput,
        *,
        duplicates: DuplicateAction = "error",
    ) -> None:
        """
        Add entities to the data set.

        Args:
            cls:
                The name of the entity class (e.g. ``"item"``).
            source:
                The input data, as an array or list of entity IDs.

                .. note::
                    Data frame support will be added in a future version.
            duplicates:
                How to handle duplicate entity IDs.
        """
        if isinstance(source, pd.DataFrame):  # pragma: nocover
            raise NotImplementedError()
        if isinstance(source, pa.Table):  # pragma: nocover
            raise NotImplementedError()

        self._validate_entity_name(cls)

        if cls not in self.schema.entities:
            self.add_entity_class(cls)

        id_name = id_col_name(cls)
        log = self._log.bind(class_name=cls)

        # we have a sequence of IDs
        ids: pa.Array = pa.array(source)

        # figure out the new schema
        schema = pa.schema({id_name: ids.type})
        table = self._tables[cls]
        if table is not None:
            schema = pa.unify_schemas([table.schema, schema], promote_options="permissive")
            if not schema.equals(table.schema):
                log.debug("upgrading existing table schema")
                table = table.cast(schema)
        elif pa.types.is_integer(ids.type):
            self.schema.entities[cls].id_type = "int"
        elif pa.types.is_string(ids.type):
            self.schema.entities[cls].id_type = "str"
        else:
            raise TypeError(f"invalid ID type {ids.type}")

        id_field = schema.field(id_name)
        id_type: pa.DataType = id_field.type

        # de-duplicate and sort the IDs
        ids = pc.unique(ids).sort()

        n = len(ids)
        if n < len(source):
            raise DataError("cannot insert duplicate entity IDs")

        if not id_type.equals(ids.type):
            log.debug("casting IDs from type %s to %s", ids.type, id_type)
            ids = ids.cast(id_type)

        if table is not None:
            vocab = self._vocabularies[cls]
            nums = vocab.numbers(ids, format="arrow", missing="null")
            fresh_ids = pc.filter(ids, pc.invert(nums.is_valid()))
            assert len(fresh_ids) == nums.null_count
        else:
            fresh_ids = ids

        if len(fresh_ids) < n and duplicates == "error":
            n_dupes = n - len(fresh_ids)
            raise DataError(f"found {n_dupes} duplicate IDs, but re-inserts not allowed")

        log.debug(
            "adding %d new IDs (%d existing)",
            len(fresh_ids),
            0 if table is None else table.num_rows,
        )
        new_tbl = pa.table({id_name: fresh_ids})
        if table is None:
            table = new_tbl
        else:
            table = pa.concat_tables([table, new_tbl], promote_options="permissive")
        log.debug("have %d entities", table.num_rows)

        self._tables[cls] = table
        self._vocabularies[cls] = Vocabulary(table.column(id_name), name=cls, reorder=False)

    def add_relationships(
        self,
        cls: str,
        data: TableInput,
        *,
        entities: RelationshipEntities | None = None,
        missing: MissingEntityAction = "error",
        allow_repeats: bool = True,
        interaction: bool | Literal["default"] = False,
        _warning_parent: int = 0,
        remove_repeats: bool | Literal["exact"] = False,
    ) -> None:
        """
        Add relationship records to the data set.

        This method adds relationship records, provided as a Pandas data frame
        or an Arrow table, to the data set being built.  The relationships can
        be of a new class (in which case it will be created), or new
        relationship records for an existing class.

        For each entity ``E`` participating in the relationship, the table must
        have a column named ``E_id`` storing the entity IDs.

        Args:
            cls:
                The name of the interaction class (e.g. ``rating``,
                ``purchase``).
            data:
                The interaction data.
            entities:
                The entity classes involved in this interaction class.
            missing:
                What to do when interactions reference nonexisting entities; can
                be ``"error"`` or ``"insert"``.
            allow_repeats:
                Whether repeated interactions are allowed.
            interaction:
                Whether this is an interaction relationship or not; can be
                ``"default"`` to indicate this is the default interaction
                relationship.
            remove_repeats:
                If ``True``, repeated interactions will be removed. If ``"exact"``,
                duplicated interactions will be removed.
        """
        if isinstance(data, pd.DataFrame):
            table = pa.Table.from_pandas(data, preserve_index=False)
        elif isinstance(data, dict):
            table = pa.table(data)  # type: ignore
        else:
            table = data

        log = self._log.bind(class_name=cls, count=table.num_rows)

        rc_def = self.schema.relationships.get(cls, None)
        if rc_def is None:
            if entities is None:
                warnings.warn(
                    f"relationship class {cls} unknown and no entities specified,"
                    " inferring from columns",
                    DataWarning,
                    stacklevel=2 + _warning_parent,
                )
                entities = [c[:-3] for c in table.column_names if c.endswith("_id")]
            self.add_relationship_class(
                cls,
                entities,
                allow_repeats=allow_repeats,
                interaction=bool(interaction),
            )
            if interaction == "default":
                self.schema.default_interaction = cls
            rc_def = self.schema.relationships[cls]

        # FIXME: remove this segment when we can make it work

        if allow_repeats and not rc_def.repeats.is_forbidden:
            rc_def.repeats = AllowableTroolean.PRESENT

        link_id_cols = set()
        link_nums = {}
        link_mask = None
        for alias, e_type in rc_def.entities.items():
            e_type = e_type or alias
            ids = table.column(id_col_name(alias))
            if missing == "insert":
                log.debug("ensuring all entities exist")
                self.add_entities(e_type, pc.unique(ids), duplicates="update")
            e_tbl = self._tables.get(e_type, None)
            if e_tbl is None:  # pragma: nocover
                raise DataError(f"no entities of class {e_type}")

            e_nums = self._resolve_entity_ids(e_type, ids, e_tbl)
            if e_nums.null_count:
                if missing == "error":
                    raise DataError(f"{e_nums.null_count} unknown IDs for entity class {e_type}")
                assert missing == "filter"
                e_valid = e_nums.is_valid()
                if link_mask is None:
                    link_mask = e_valid
                else:
                    link_mask = pc.and_(link_mask, e_valid)
            link_nums[num_col_name(alias)] = e_nums
            link_id_cols.add(id_col_name(alias))

        new_table = pa.table(link_nums)

        for col in table.column_names:
            if col not in link_id_cols:
                new_table = new_table.append_column(col, table.column(col))
                if col not in rc_def.attributes:
                    self._validate_attribute_name(col)
                    rc_def.attributes[col] = ColumnSpec()

        if link_mask is not None:
            log.debug("filtering links to known entities")
            new_table = new_table.filter(link_mask)
        log.debug("adding %d new rows", new_table.num_rows)

        cur_table = self._tables[cls]

        # combine the tables to save them
        incoming_table = new_table
        if cur_table is not None:
            new_table = pa.concat_tables([cur_table, new_table], promote_options="permissive")

        if remove_repeats is True:
            new_table = self._remove_repeated_relationships(t=new_table, rel=rc_def)
        else:
            if remove_repeats == "exact":
                new_table = self._remove_duplicated_relationships(t=new_table)
                self._rel_coords[cls] = None

            if rc_def.repeats.is_forbidden:
                _log.debug("checking for repeated interactions")

                # FIXME: the coordinate table preservation will be out-of-order with
                # respect to sorted tables. This will not affect correctness for the
                # purposes of checking if duplicates exist, but will cause problems
                # if we count on locations for other purposes.  It also can increase
                # memory use, since the coordinate table keeps a reference to the
                # pre-sorted data.

                # get the coordinate table for the previously-existing entries
                coords = self._rel_coords.get(cls, None)
                if coords is None:
                    coords = _data_accel.CoordinateTable(len(rc_def.entities))
                    if cur_table is not None:
                        _log.debug("building coord table from previous interactions")
                        coo = cur_table.select(link_nums.keys())
                        coords.extend(coo.to_batches())
                        assert len(coo) == len(cur_table)
                    self._rel_coords[cls] = coords

                # add the new data
                coords.extend(incoming_table.select(link_nums.keys()).to_batches())
                n_dupes = len(coords) - coords.unique_count()
                _log.debug(
                    "coordinates: %d total, %d unique, %d dupe",
                    len(coords),
                    coords.unique_count(),
                    n_dupes,
                )

                if n_dupes:
                    if rc_def.repeats.is_allowed:
                        _log.debug("found %d repeat interactions", n_dupes)
                        rc_def.repeats = AllowableTroolean.PRESENT
                    else:
                        _log.error(
                            "found %d forbidden repeat interactions (of %d)",
                            n_dupes,
                            new_table.num_rows,
                        )
                        raise DataError(
                            f"repeated interactions not allowed for relationship class {cls}"
                        )

        log.debug(
            "saving new relationship table", total_rows=new_table.num_rows, schema=new_table.schema
        )
        self._tables[cls] = new_table

    def add_interactions(
        self,
        cls: str,
        data: TableInput,
        *,
        entities: RelationshipEntities | None = None,
        missing: MissingEntityAction = "error",
        allow_repeats: bool = True,
        default: bool = False,
        remove_repeats: bool | Literal["exact"] = False,
    ) -> None:
        """
        Add a interaction records to the data set.

        This method adds new interaction records, provided as a Pandas data
        frame or an Arrow table, to the data set being built.  The interactions
        can be of a new class (in which case it will be created), or new
        interactions for an existing class.

        For each entity ``E`` participating in the interaction, the table must
        have a column named ``E_id`` storing the entity IDs.

        Interactions should usually have user as the first entity and item as
        the last; the default interaction matrix logic uses the first and last
        entities as the rows and columns, respectively, of the interaction
        matrix.


        Args:
            cls:
                The name of the interaction class (e.g. ``rating``,
                ``purchase``).
            data:
                The interaction data.
            entities:
                The entity classes involved in this interaction class.
            missing:
                What to do when interactions reference nonexisting entities; can
                be ``"error"`` or ``"insert"``.
            allow_repeats:
                Whether repeated interactions are allowed.
            default:
                If ``True``, set this as the default interaction class (if the
                dataset has more than one interaction class).
            remove_repeats:
                If ``True``, repeated interactions will be removed. If ``"exact"``,
                duplicated interactions will be removed.
        """
        self.add_relationships(
            cls,
            data,
            entities=entities,
            missing=missing,
            allow_repeats=allow_repeats,
            interaction="default" if default else True,
            _warning_parent=1,
            remove_repeats=remove_repeats,
        )

    def filter_interactions(
        self,
        cls: str,
        min_time: int | float | dt.datetime | None = None,
        max_time: int | float | dt.datetime | None = None,
        remove: pa.Table | dict[str, ArrayLike] | pd.DataFrame | None = None,
    ):
        """
        Filter interactions based on timestamp or to remove particular entities.

        Args:
            cls:
                The interaction class to filter.
            min_time:
                The minimum interaction time to keep (inclusive).
            max_time:
                The maximum interaction time to keep (exclusive).
            remove:
                Combinations of entity numbers or IDs to remove.  The entities
                are filtered using an anti-join with this table, so providing a
                single column of entity IDs or numbers will remove all
                interactions associated with the listed entities.
        """
        tbl = self._tables[cls]
        if tbl is None:  # pragma: nocover
            raise ValueError(f"interaction class {cls} is empty")

        if min_time is not None or max_time is not None:
            if "timestamp" not in tbl.column_names:
                raise RuntimeError("timestamp column required to filter by timestamp")
            schema = tbl.schema
            ts_field = schema.field("timestamp")

        mask = None
        if min_time is not None:
            min_time = _conform_time(min_time, ts_field.type)
            mask = pc.greater_equal(tbl.column("timestamp"), min_time)
        if max_time is not None:
            max_time = _conform_time(max_time, ts_field.type)
            mask2 = pc.less(tbl.column("timestamp"), max_time)

            if mask is None:
                mask = mask2
            else:
                mask = pc.and_(mask, mask2)

        if mask is not None:
            tbl = tbl.filter(mask)

        if remove is not None:
            if isinstance(remove, pd.DataFrame):
                remove = pa.Table.from_pandas(remove, preserve_index=False)
            else:
                remove = pa.table(remove)  # type: ignore
            assert isinstance(remove, pa.Table)
            rtbl_cols = {}
            for cname in remove.column_names:
                if cname.endswith("_id"):
                    ent = cname[:-3]
                    num_col = num_col_name(ent)
                    etbl = self._tables[ent]
                    assert etbl is not None
                    col = remove.column(cname)
                    id_col = etbl.column(cname)
                    nums = pc.index_in(col, id_col)
                    rtbl_cols[num_col] = nums
                elif cname.endswith("_num"):
                    rtbl_cols[cname] = remove.column(cname)
                else:  # pragma: nocover
                    raise ValueError(f"invalid removal column {cname}")

            remove = pa.table(rtbl_cols)
            tbl = tbl.join(remove, remove.column_names, join_type="left anti")

        self._tables[cls] = tbl
        self._rel_coords[cls] = None

    def binarize_ratings(
        self,
        cls: str = "rating",
        min_pos_rating: float = 3.0,
        method: str = "remove",
    ):
        """
        Binarize the ratings in a relationship class.

        Args:
            cls : The relationship class to binarize (default: "rating").
            min_pos_rating : Minimum rating to consider as positive.
            method: 'zero' to set ratings to 0/1, 'remove' to drop rows below min_rating.
        """
        tbl = self._tables.get(cls, None)
        if tbl is None:
            raise ValueError(f"relationship class {cls} is empty")

        if "rating" not in tbl.column_names:
            raise ValueError(f"relationship class {cls} does not have a 'rating' column")

        min_rating = pa.scalar(min_pos_rating, tbl.column("rating").type)
        rating_col = tbl.column("rating")
        if not (
            pc.min(tbl.column("rating")).as_py()
            <= min_pos_rating
            <= pc.max(tbl.column("rating")).as_py()
        ):
            raise ValueError(
                f"min_pos_rating {min_pos_rating} is not in the range of ratings "
                f"[{pc.min(tbl.column('rating')).as_py()}, {pc.max(tbl.column('rating')).as_py()}]"
            )

        if method == "zero":
            # Set rating to 1 if >= min_rating, else 0
            mask = pc.greater_equal(tbl.column("rating"), min_rating)
            binarized = pc.cast(mask, rating_col.type)
            tbl = tbl.set_column(tbl.schema.get_field_index("rating"), "rating", binarized)
        elif method == "remove":
            # Keep only rows where rating >= min_rating
            mask = pc.greater_equal(tbl.column("rating"), min_rating)
            tbl = tbl.filter(mask)
        else:
            raise ValueError("method must be 'zero' or 'remove'")

        self._tables[cls] = tbl
        self._rel_coords[cls] = None

    def clear_relationships(self, cls: str):
        """
        Remove all records for a specified relationship class.
        """
        self._tables[cls] = _empty_rel_table(self.schema.relationships[cls].entity_class_names)
        self._rel_coords[cls] = None

    @overload
    def add_scalar_attribute(
        self,
        cls: str,
        name: str,
        data: pd.Series[Any] | TableInput,
        /,
        *,
        dictionary: bool = False,
    ) -> None: ...
    @overload
    def add_scalar_attribute(
        self,
        cls: str,
        name: str,
        entities: IDSequence | tuple[IDSequence, ...],
        values: ArrayLike,
        /,
        *,
        dictionary: bool = False,
    ) -> None: ...
    def add_scalar_attribute(
        self,
        cls: str,
        name: str,
        entities: IDSequence | tuple[IDSequence, ...] | pd.Series[Any] | TableInput,
        values: ArrayLike | None = None,
        *,
        dictionary: bool = False,
    ) -> None:
        """
        Add a scalar attribute to an entity class.

        Args:
            cls:
                The entity class name.
            name:
                The attribute name.
            entities:
                The IDs for the entities whose attribute should be set.
            values:
                The attribute values.
            data:
                A Pandas datatframe or Arrow table storing entity IDs and
                attribute values.
            dictionary:
                ``True`` to dictionary-encode the attribute values (saves space
                for string categorical values).
        """
        if name in self.schema.entities[cls].attributes:  # pragma: nocover
            raise NotImplementedError("updating or replacing existing attributes not supported")
        self._validate_attribute_name(name)

        id_col = id_col_name(cls)

        if values is None:
            if isinstance(entities, pd.Series):
                # it is a series, use the index
                values = entities.values
                entities = entities.index.values
            elif isinstance(entities, pd.DataFrame):
                values = entities[name].values
                if id_col in entities.columns:
                    entities = entities[id_col].values
                else:
                    entities = entities.index.values

        e_tbl = self._tables[cls]
        if e_tbl is None:  # pragma: nocover
            raise DataError(f"no entities of class {cls}")

        nums = self._resolve_entity_ids(cls, entities, e_tbl)
        if not np.all(nums.is_valid()):  # pragma: nocover
            n_bad = nums.is_valid().sum().as_py()
            raise DataError(f"{n_bad} unknown entity IDs")

        val_array: pa.Array = pa.array(values)  # type: ignore
        nums_np = nums.to_numpy()
        vals_np = np.full(e_tbl.num_rows, None, dtype=object)
        vals_np[nums_np] = val_array.to_numpy(zero_copy_only=False)
        val_col = pa.array(vals_np, type=val_array.type)

        if dictionary:
            val_col = pc.dictionary_encode(val_col)

        self._tables[cls] = e_tbl.append_column(name, val_col)
        self.schema.entities[cls].attributes[name] = ColumnSpec(layout=AttrLayout.SCALAR)

    @overload
    def add_list_attribute(
        self,
        cls: str,
        name: str,
        data: pd.Series[Any] | TableInput,
        /,
        *,
        dictionary: bool = False,
    ) -> None: ...
    @overload
    def add_list_attribute(
        self,
        cls: str,
        name: str,
        entities: IDSequence | tuple[IDSequence, ...],
        values: ArrayLike,
        /,
        *,
        dictionary: bool = False,
    ) -> None: ...
    def add_list_attribute(
        self,
        cls: str,
        name: str,
        entities: IDSequence | tuple[IDSequence, ...] | pd.Series[Any] | TableInput,
        values: ArrayLike | None = None,
        *,
        dictionary: bool = False,
    ) -> None:
        """
        Add a list attribute to an entity class.

        Args:
            cls:
                The entity class name.
            name:
                The attribute name.
            entities:
                The IDs for the entities whose attribute should be set.
            values:
                The attribute values (an array or list of lists)
            data:
                A Pandas datatframe or Arrow table storing entity IDs and
                attribute values.
            dictionary:
                ``True`` to dictionary-encode the attribute values (saves space
                for string categorical values).
        """
        if name in self.schema.entities[cls].attributes:  # pragma: nocover
            raise NotImplementedError("updating or replacing existing attributes not supported")
        self._validate_attribute_name(name)

        id_col = id_col_name(cls)

        if values is None:
            if isinstance(entities, pd.Series):
                # it is a series, use the index
                values = entities.values
                entities = entities.index.values
            elif isinstance(entities, pd.DataFrame):
                values = entities[name].values
                if id_col in entities.columns:
                    entities = entities[id_col].values
                else:
                    entities = entities.index.values

        e_tbl = self._tables[cls]
        if e_tbl is None:  # pragma: nocover
            raise DataError(f"no entities of class {cls}")
        nums = self._resolve_entity_ids(cls, entities, e_tbl)
        if not np.all(nums.is_valid()):  # pragma: nocover
            n_bad = nums.is_valid().sum().as_py()
            raise DataError(f"{n_bad} unknown entity IDs")

        val_array: pa.Array = pa.array(values)  # type: ignore
        if not pa.types.is_list(val_array.type):
            raise TypeError("attribute data did not resolve to list")

        if isinstance(val_array, pa.ChunkedArray):  # pragma: nocover
            val_array = val_array.combine_chunks()
        assert isinstance(val_array, pa.ListArray)

        if dictionary:
            val_array = pa.ListArray.from_arrays(
                val_array.offsets, pc.dictionary_encode(val_array.values)
            )

        nums = nums.to_numpy()

        # we have to do surgery on the offsets and values
        val_col = _expand_and_align_list_array(e_tbl.num_rows, nums, val_array)

        self._tables[cls] = e_tbl.append_column(name, val_col)
        self.schema.entities[cls].attributes[name] = ColumnSpec(layout=AttrLayout.LIST)

    def add_vector_attribute(
        self,
        cls: str,
        name: str,
        entities: IDSequence | tuple[IDSequence, ...],
        values: pa.Array[Any] | pa.ChunkedArray[Any] | np.ndarray[tuple[int, int], Any] | sparray,
        /,
        dim_names: ArrayLike | pd.Index[Any] | Sequence[Any] | None = None,
    ) -> None:
        """
        Add a vector attribute to a set of entities.

        .. warning::

            Dense vector attributes are stored densely, even for entities for
            which it is not set. High-dimensional vectors can therefore take up
            a lot of space.

        Args:
            cls:
                The entity class name.
            name:
                The attribute name.
            entities:
                The entity IDs to which the attribute should be attached.
            values:
                The attribute values, as a fixed-length list array or a
                two-dimensional NumPy array (for dense vector attributes) or a
                SciPy sparse array (for sparse vector attributes).
            dim_names:
                The names for the dimensions of the array.
        """
        if name in self.schema.entities[cls].attributes:  # pragma: nocover
            raise NotImplementedError("updating or replacing existing attributes not supported")
        self._validate_attribute_name(name)

        e_tbl = self._tables[cls]
        if e_tbl is None:  # pragma: nocover
            raise DataError(f"no entities of class {cls}")
        nums = self._resolve_entity_ids(cls, entities, e_tbl)
        if not np.all(nums.is_valid()):  # pragma: nocover
            n_bad = nums.is_valid().sum().as_py()
            raise DataError(f"{n_bad} unknown entity IDs")

        tbl_valid = np.zeros(e_tbl.num_rows, dtype=np.bool_)
        tbl_valid[nums.to_numpy()] = True

        metadata = {}
        if dim_names is not None:
            metadata["lenskit:names"] = json.dumps(np.asarray(dim_names).tolist())

        if isinstance(values, sparray):
            vec_col = self._add_sparse_vector_attribute(cls, name, values, nums, e_tbl, tbl_valid)
        elif isinstance(values, np.ndarray):
            vec_col = self._add_dense_vector_attribute_numpy(
                cls, name, values, nums, e_tbl, tbl_valid
            )
        else:
            vec_col = self._add_dense_vector_attribute(cls, name, values, nums, e_tbl, tbl_valid)

        field = pa.field(name, vec_col.type, nullable=True, metadata=metadata)
        self._tables[cls] = e_tbl.append_column(field, vec_col)

    def _validate_attribute_name(self, attribute_name: str):
        if attribute_name.endswith(("_id", "_num")) or attribute_name.startswith("_"):
            raise ValueError(f"invalid attribute name {attribute_name}")

    def _add_sparse_vector_attribute(
        self,
        cls: str,
        name: str,
        values: sparray,
        rows: pa.Int32Array,
        table: pa.Table,
        valid: NDArray[np.bool_],
    ):
        csr: csr_array = values.tocsr()  # type: ignore
        assert isinstance(csr, csr_array)
        csr.sort_indices()

        array = SparseRowArray.from_scipy(csr)
        vec_col = _expand_and_align_list_array(table.num_rows, rows.to_numpy(), array.storage)
        vec_col = SparseRowArray.from_array(vec_col)
        self.schema.entities[cls].attributes[name] = ColumnSpec(
            layout=AttrLayout.SPARSE, vector_size=array.dimension
        )
        return vec_col

    def _add_dense_vector_attribute(
        self,
        cls: str,
        name: str,
        values: pa.Array[Any] | pa.ChunkedArray[Any],
        rows: pa.Int32Array,
        table: pa.Table,
        valid: NDArray[np.bool_],
    ):
        if isinstance(values, pa.ChunkedArray):
            values = values.combine_chunks()
        if not pa.types.is_fixed_size_list(values.type):
            raise TypeError("attribute data must be fixed-size list")
        assert isinstance(values, pa.FixedSizeListArray)
        v_valid = values.is_valid().to_numpy(zero_copy_only=False)

        # no nulls: use as-is
        if np.all(valid) and np.all(v_valid):
            return values

        # find the rows where we have a valid column value
        c_valid = valid.copy()
        c_valid[rows] &= v_valid

        # nulls: we actually need a list array for sparsity + storage
        size = values.type.list_size
        values = values.cast(pa.list_(values.type.value_type))
        col = _expand_and_align_list_array(table.num_rows, rows.to_numpy(), values)

        # sizes = np.zeros(table.num_rows + 1, dtype=np.int32)
        # sizes[1:][c_valid] = size
        # offsets = np.cumsum(sizes, dtype=np.int32)
        # assert offsets[-1] == size * np.sum(c_valid)

        # # We need to reorder to match the order in the original table.  The
        # # argsort gives give us the order in which we need to consult rows in
        # # the matrix, but realigned to match the order in the table, not the
        # # order in the source vector.
        # ipos = np.argsort(rows)
        # if np.any(np.diff(ipos) < 0):
        #     values = values.take(ipos)

        # vfilt = values.filter(v_valid)
        # assert len(vfilt) == np.sum(c_valid)

        # col = pa.ListArray.from_arrays(pa.array(offsets), vfilt.values, mask=pa.array(~c_valid))
        assert np.all(col.is_valid().to_numpy(zero_copy_only=False) == c_valid)
        self.schema.entities[cls].attributes[name] = ColumnSpec(
            layout=AttrLayout.VECTOR, vector_size=size
        )
        return col

    def _add_dense_vector_attribute_numpy(
        self,
        cls: str,
        name: str,
        values: NDArray[Any],
        rows: pa.Int32Array,
        table: pa.Table,
        valid: NDArray[np.bool_],
    ):
        if len(values.shape) != 2:
            raise TypeError("values must be 2D array")
        nrow, ncol = values.shape
        arr = pa.FixedSizeListArray.from_arrays(values.ravel(), ncol)
        return self._add_dense_vector_attribute(cls, name, arr, rows, table, valid)

    def build(self) -> Dataset:
        """
        Build the dataset.
        """
        return Dataset(self.build_container())

    def build_container(self) -> DataContainer:
        """
        Build a data container (backing store for a dataset).
        """
        log = _log.bind(name=self.name)
        log.debug("assembling data container")
        tables = {}
        for n, t in self._tables.items():
            if t is None:
                log.debug("fabricating empty table %s", n)
                tables[n] = pa.table({id_col_name(n): pa.array([], type=pa.int64())})
            else:
                rel = self.schema.relationships.get(n, None)
                if rel is not None:
                    # currently not used due to coordinate table
                    # self._check_repeat_interactions(t, rel)
                    if rel.repeats.is_forbidden and len(rel.entities) == 2:
                        e_cols = [e + "_num" for e in rel.entities.keys()]
                        if not _data_accel.is_sorted_coo(t.to_batches(), *e_cols):
                            log.debug("sorting non-repeating relationship %s", n)
                            t = t.sort_by([(c, "ascending") for c in e_cols])

                tables[n] = t

        return DataContainer(self.schema.model_copy(), tables, _rel_coords=self._rel_coords)

    def save(self, path: str | PathLike[str]):
        """
        Save the dataset to disk in the LensKit native format.

        Args:
            path:
                The path where the dataset will be saved (will be created as a
                directory).
        """
        container = self.build_container()
        container.save(path)

    def _remove_duplicated_relationships(self, t: pa.Table) -> pa.Table:
        temp_df = t.to_pandas()
        temp_df.drop_duplicates(inplace=True)
        t = pa.Table.from_pandas(temp_df, preserve_index=False)
        return t

    def _remove_repeated_relationships(self, t: pa.Table, rel: RelationshipSchema) -> pa.Table:
        if "timestamp" in t.column_names:
            t = t.sort_by([("timestamp", "ascending")])
        t_modified = t.add_column(0, "_row_number", pa.array(np.arange(t.num_rows)))
        t_modified = t_modified.group_by(
            [entity + "_num" for entity in rel.entity_class_names]
        ).aggregate([("_row_number", "max")])
        t_modified = t_modified.sort_by([("_row_number_max", "ascending")])
        t = t.take(t_modified.column("_row_number_max"))
        return t

    def _validate_entity_name(self, entity_name: str):
        if entity_name.startswith("_"):
            raise ValueError(f"invalid entity name {entity_name}")

    def _resolve_entity_ids(
        self, cls: str, ids: IDSequence, table: pa.Table | None = None
    ) -> pa.Int32Array:
        if isinstance(ids, pa.ChunkedArray):
            tgt_ids = ids.combine_chunks()
        else:
            tgt_ids = pa.array(ids)  # type: ignore
        vocab = self._vocabularies.get(cls, None)
        if vocab is None:
            return pa.nulls(len(tgt_ids), type=pa.int32())
        return vocab.numbers(tgt_ids, format="arrow", missing="null")


def _expand_and_align_list_array(
    out_len: int, rows: NDArray[np.int32], lists: pa.ListArray
) -> pa.ListArray:
    """
    Expand a list array, so that its lists are on the specified rows of the
    output array, with other entries null.
    """
    assert pc.count_distinct(rows).as_py() == len(rows)
    assert len(lists) == len(rows)
    # only valid inputs
    valid = lists.is_valid().to_numpy(zero_copy_only=False)
    if not np.all(valid):
        rows = rows[valid]
        lists = lists.drop_null()
        assert len(lists) == len(rows)

    # reorder input to align with the output
    order = np.argsort(rows)
    if np.any(np.diff(order) < 0):
        lists = lists.take(order)
        rows = rows[order]

    # now we do surgery — put the lengths into place, and set up offsets
    sizes = np.zeros(out_len + 1, dtype=np.int32)
    sizes[rows + 1] = lists.value_lengths().to_numpy()
    offsets = np.cumsum(sizes, dtype=np.int32)
    offsets = pa.array(offsets)

    mask = np.ones(out_len, dtype=np.bool_)
    mask[rows] = False
    mask = pa.array(mask)

    # we can now construct the array — these offsets point into the source array
    return pa.ListArray.from_arrays(offsets, lists.values, mask=mask)


def _empty_rel_table(types: list[str]) -> pa.Table:
    return pa.table({num_col_name(t): pa.array([], pa.int32()) for t in types})


def _conform_time(time: int | float | str | dt.datetime, col_type: pa.DataType):
    if isinstance(time, str):
        time = dt.datetime.fromisoformat(time)

    if pa.types.is_timestamp(col_type):
        if not isinstance(time, dt.datetime):
            return dt.datetime.fromtimestamp(time)
    elif isinstance(time, dt.datetime):
        return time.timestamp()

    return time
