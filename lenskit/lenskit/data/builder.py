# pyright: basic
from __future__ import annotations

import datetime as dt
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
from scipy.sparse import sparray

from lenskit.diagnostics import DataError, DataWarning
from lenskit.logging import get_logger

from .container import DataContainer
from .dataset import Dataset
from .schema import (
    AllowableTroolean,
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
        else:
            self.schema = DataSchema(name=name, entities={"item": EntitySchema()})
            self._tables = {"item": None}

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

    def add_relationships(
        self,
        cls: str,
        data: TableInput,
        *,
        entities: RelationshipEntities | None = None,
        missing: MissingEntityAction = "error",
        allow_repeats: bool = True,
        interaction: bool | Literal["default"] = False,
    ) -> None:
        if isinstance(data, pd.DataFrame):
            table = pa.Table.from_pandas(data)
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
                    stacklevel=2,
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

            e_ids = e_tbl.column(id_col_name(e_type))
            e_nums = pc.index_in(ids, e_ids)
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
                Combinations of entity numbers (**not** IDs) to remove.
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
            remove = pa.table(remove)  # type: ignore
            assert isinstance(remove, pa.Table)
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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def add_vector_attribute(
        self,
        cls: str,
        name: str,
        entities: IDSequence | tuple[IDSequence, ...],
        values: ArrayLike | sparray,
        /,
        dims: ArrayLike | pd.Index[Any] | Sequence[Any] | None = None,
    ) -> None: ...

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
