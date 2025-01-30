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
NPArray1D: TypeAlias = np.ndarray[type[int], np.dtype[NPT]]

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
    """

    schema: DataSchema
    """
    The data schema assembled so far.  Do not modify this schema directly.
    """

    _log: structlog.stdlib.BoundLogger
    _tables: dict[str, pa.Table | None]
    _indexes: dict[str, pd.Index]

    def __init__(self, name: str | DataContainer | Dataset | None = None):
        """
        Create a new dataset builder.

        Args:
            name:
                The dataset name. Can also be a data container or a dataset,
                which will initialize this builder with its contents to extend
                or modify.
        """
        if isinstance(name, Dataset):
            name._ensure_loaded()
            name = name._data
        if isinstance(name, DataContainer):
            self.schema = name.schema.model_copy()
            self._tables = {n: t for (n, t) in name.tables.items()}
            self._indexes = {
                n: pd.Index(name.tables[n].column(id_col_name(n)).to_numpy(zero_copy_only=False))
                for n in name.schema.entities.keys()
            }
        else:
            self.schema = DataSchema(name=name, entities={"item": EntitySchema()})
            self._tables = {"item": None}
            self._indexes = {}

        self._log = _log.bind(ds_name=name)

    @property
    def name(self) -> str | None:
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
        if name in self._tables:
            raise ValueError(f"class name “{name}” already in use")

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
        if name in self._tables:
            raise ValueError(f"class name “{name}” already in use")

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
        if isinstance(source, pd.DataFrame):  # pragma: nocover
            raise NotImplementedError()
        if isinstance(source, pa.Table):  # pragma: nocover
            raise NotImplementedError()

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
            col = table.column(id_name)
            is_known = pc.is_in(ids, col)
            fresh_ids = pc.filter(ids, pc.invert(is_known))
        else:
            fresh_ids = ids

        if len(fresh_ids) < n and duplicates == "error":
            n_dupes = n - len(fresh_ids)
            raise DataError(f"found {n_dupes} duplicate IDs, but re-inserts not allowed")

        log.debug("adding %d new IDs", len(fresh_ids))
        new_tbl = pa.table({id_name: fresh_ids})
        if table is None:
            table = new_tbl
        else:
            table = pa.concat_tables([table, new_tbl], promote_options="permissive")

        self._tables[cls] = table
        self._indexes[cls] = pd.Index(table.column(id_name).to_numpy(zero_copy_only=False))

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
    ) -> None:
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
                cls, entities, allow_repeats=allow_repeats, interaction=bool(interaction)
            )
            if interaction == "default":
                self.schema.default_interaction = cls
            rc_def = self.schema.relationships[cls]

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
            e_valid = e_nums.is_valid()
            if not pc.all(e_valid).as_py():
                if missing == "error":
                    n_bad = len(e_nums) - pc.sum(e_valid).as_py()  # type: ignore
                    raise DataError(f"{n_bad} unknown IDs for entity class {e_type}")
                assert missing == "filter"
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

        if link_mask is not None:
            log.debug("filtering links to known entities")
            new_table = new_table.filter(link_mask)
        log.debug("adding %d new rows", new_table.num_rows)

        if "count" in new_table.column_names:  # pragma: nocover
            raise NotImplementedError("count attributes are not yet implemented")

        cur_table = self._tables[cls]
        if cur_table is not None:
            new_table = pa.concat_tables([cur_table, new_table], promote_options="permissive")

        if not rc_def.repeats.is_present:
            _log.debug("checking for repeated interactions")
            # we have to bounce to pandas for multi-column duplicate detection
            ndf = new_table.select(list(link_nums.keys())).to_pandas()
            dupes = ndf.duplicated()
            if np.any(dupes):
                if rc_def.repeats.is_allowed:
                    rc_def.repeats = AllowableTroolean.PRESENT
                else:
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
    ) -> None:
        self.add_relationships(
            cls,
            data,
            entities=entities,
            missing=missing,
            allow_repeats=allow_repeats,
            interaction="default" if default else True,
            _warning_parent=1,
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

    def clear_relationships(self, cls: str):
        self._tables[cls] = _empty_rel_table(self.schema.relationships[cls].entity_class_names)

    @overload
    def add_scalar_attribute(
        self, cls: str, name: str, data: pd.Series[Any] | TableInput, /
    ) -> None: ...
    @overload
    def add_scalar_attribute(
        self,
        cls: str,
        name: str,
        entities: IDSequence | tuple[IDSequence, ...],
        values: ArrayLike,
        /,
    ) -> None: ...
    def add_scalar_attribute(
        self,
        cls: str,
        name: str,
        entities: IDSequence | tuple[IDSequence, ...] | pd.Series[Any] | TableInput,
        values: ArrayLike | None = None,
    ) -> None:
        if name in self.schema.entities[cls].attributes:  # pragma: nocover
            raise NotImplementedError("updating or replacing existing attributes not supported")

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
        tbl_mask = np.zeros(e_tbl.num_rows, dtype=np.bool_)
        tbl_mask[nums.to_numpy()] = True
        tbl_mask = pa.array(tbl_mask)
        val_col = pa.nulls(e_tbl.num_rows, val_array.type)
        val_col = pc.replace_with_mask(val_col, tbl_mask, val_array)

        self._tables[cls] = e_tbl.append_column(name, val_col)
        self.schema.entities[cls].attributes[name] = ColumnSpec(layout=AttrLayout.SCALAR)

    @overload
    def add_list_attribute(
        self, cls: str, name: str, data: pd.Series[Any] | TableInput, /
    ) -> None: ...
    @overload
    def add_list_attribute(
        self,
        cls: str,
        name: str,
        entities: IDSequence | tuple[IDSequence, ...],
        values: ArrayLike,
        /,
    ) -> None: ...
    def add_list_attribute(
        self,
        cls: str,
        name: str,
        entities: IDSequence | tuple[IDSequence, ...] | pd.Series[Any] | TableInput,
        values: ArrayLike | None = None,
    ) -> None:
        if name in self.schema.entities[cls].attributes:  # pragma: nocover
            raise NotImplementedError("updating or replacing existing attributes not supported")

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

        nums = nums.to_numpy()
        tbl_valid = np.zeros(e_tbl.num_rows, dtype=np.bool_)
        tbl_valid[nums] = True
        tbl_valid = pa.array(tbl_valid)

        # we have to do surgery on the offsets and values
        lengths = np.zeros(e_tbl.num_rows + 1, dtype=np.int32)
        lengths[nums + 1] = val_array.value_lengths().fill_null(0).to_numpy()
        offsets = np.cumsum(lengths, dtype=np.int32)

        val_col = pa.ListArray.from_arrays(
            pa.array(offsets), val_array.values, mask=pc.invert(tbl_valid)
        )

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

            The vector is stored densely, even for entities for which it is not
            set. High-dimensional vectors can therefore take up a lot of space.

        Args:
            cls:
                The entity class name.
            name:
                The attribute name.
            entities:
                The entity IDs to which the attribute should be attached.
            values:
                The attribute values, as a fixed-length list array or a
                two-dimensional NumPy array.
            dim_names:
                The names for the dimensions of the array.
        """
        if name in self.schema.entities[cls].attributes:  # pragma: nocover
            raise NotImplementedError("updating or replacing existing attributes not supported")

        e_tbl = self._tables[cls]
        if e_tbl is None:  # pragma: nocover
            raise DataError(f"no entities of class {cls}")
        nums = self._resolve_entity_ids(cls, entities, e_tbl)
        if not np.all(nums.is_valid()):  # pragma: nocover
            n_bad = nums.is_valid().sum().as_py()
            raise DataError(f"{n_bad} unknown entity IDs")

        tbl_mask = np.ones(e_tbl.num_rows, dtype=np.bool_)
        tbl_mask[nums.to_numpy()] = False

        vec_col = None
        matrix = None

        metadata = {}
        if dim_names is not None:
            metadata["lenskit:names"] = json.dumps(np.asarray(dim_names).tolist())

        if isinstance(values, sparray):
            csr: csr_array = values.tocsr()  # type: ignore

            rlen = np.zeros(e_tbl.num_rows + 1, csr.dtype)
            rlen[nums.to_numpy() + 1] = np.diff(csr.indptr)
            rowptr = np.cumsum(rlen, dtype=np.int32)

            obs_struct = pa.StructArray.from_arrays(
                [pa.array(csr.indices), pa.array(csr.data)], ["index", "value"]
            )
            vec_col = pa.ListArray.from_arrays(
                pa.array(rowptr), obs_struct, mask=pa.array(tbl_mask)
            )
            metadata["lenskit:ncol"] = str(csr.shape[1])  # type: ignore

        elif isinstance(values, pa.ChunkedArray):
            if not pa.types.is_fixed_size_list(values.type):
                raise TypeError("attribute data must be fixed-size list")

            start = 0
            for chunk in values.chunks:
                assert isinstance(chunk, pa.FixedSizeListArray)
                cnp = chunk.values.to_numpy()
                if matrix is None:
                    matrix = np.full((e_tbl.num_rows, chunk.type.list_size), np.nan, cnp.dtype)
                end = start + len(chunk)
                matrix[nums[start:end], :] = cnp.reshape((end - start, chunk.type.list_size))
                start = end

        elif isinstance(values, pa.Array):
            if not pa.types.is_fixed_size_list(values.type):
                raise TypeError("attribute data must be fixed-size list")
            assert isinstance(values, pa.FixedSizeListArray)
            vnp = values.values.to_numpy()
            matrix = np.full((e_tbl.num_rows, values.type.list_size), np.nan, vnp.dtype)
            matrix[nums] = vnp.reshape((len(nums), values.type.list_size))

        else:
            if len(values.shape) != 2:
                raise TypeError("values must be 2D array")
            nrow, ncol = values.shape
            if nrow != len(entities):
                raise ValueError("entity list and value row count must match")
            matrix = np.full((e_tbl.num_rows, ncol), np.nan, values.dtype)
            matrix[nums] = values

        if vec_col is None:
            assert matrix is not None

            tbl_mask = pa.array(tbl_mask)
            vec_col = pa.FixedSizeListArray.from_arrays(
                pa.array(matrix.ravel()), matrix.shape[1], mask=tbl_mask
            )
            self.schema.entities[cls].attributes[name] = ColumnSpec(layout=AttrLayout.VECTOR)
        else:
            self.schema.entities[cls].attributes[name] = ColumnSpec(layout=AttrLayout.SPARSE)

        field = pa.field(name, vec_col.type, nullable=True, metadata=metadata)
        self._tables[cls] = e_tbl.append_column(field, vec_col)

    def build(self) -> Dataset:
        return Dataset(self.build_container())

    def build_container(self) -> DataContainer:
        tables = {}
        for n, t in self._tables.items():
            if t is None:
                tables[n] = pa.table({id_col_name(n): pa.array([], type=pa.int64())})
            else:
                tables[n] = t

        return DataContainer(self.schema.model_copy(), tables)

    def save(self, path: str | PathLike[str]):
        """
        Save the dataset to disk in the LensKit native format.

        Args:
            path:
                The path where the dataset will be saved (will be created as a
                directory)
        """
        container = self.build_container()
        container.save(path)

    def _resolve_entity_ids(
        self, cls: str, ids: IDSequence, table: pa.Table | None = None
    ) -> pa.Int32Array:
        tgt_ids = np.array(ids)  # type: ignore
        index = self._indexes.get(cls, None)
        if index is None:
            return pa.nulls(len(tgt_ids), type=pa.int32())
        nums = np.require(index.get_indexer_for(tgt_ids), np.int32)
        return pc.if_else(nums >= 0, nums, None)


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
