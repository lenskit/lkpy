# pyright: basic
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import structlog
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import sparray

from lenskit.data.vocab import Vocabulary
from lenskit.diagnostics import DataError, DataWarning
from lenskit.logging import get_logger

from .dataset import Dataset
from .matrix import MatrixDataset
from .schema import AllowableTroolean, DataSchema, EntitySchema, RelationshipSchema, check_name
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

    def __init__(self, name: str | None = None):
        self.schema = DataSchema(name=name, entities={"item": EntitySchema()})
        self._log = _log.bind(ds_name=name)
        self._tables = {"item": None}

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

        return tbl.field(_id_name(name)).type

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

        self.schema.relationships[name] = RelationshipSchema(
            entities=e_dict,
            interaction=interaction,
            repeats=AllowableTroolean.ALLOWED if allow_repeats else AllowableTroolean.FORBIDDEN,
        )
        self._tables[name] = None

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
        if isinstance(source, pd.DataFrame):
            raise NotImplementedError()
        if isinstance(source, pa.Table):
            raise NotImplementedError()

        if cls not in self.schema.entities:
            self.add_entity_class(cls)

        id_name = _id_name(cls)
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

        id_field = schema.field_by_name(id_name)
        id_type: pa.DataType = id_field.type

        # de-duplicate and sort the IDs
        ids = pc.unique(ids)
        ids = pc.take(ids, pc.sort_indices(ids))

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
        if table is None:
            table = pa.table({id_name: fresh_ids})
        else:
            new_batch = pa.record_batch({id_name: fresh_ids})
            new_batch = new_batch.cast(schema)
            batches = table.to_batches()
            batches.append(new_batch)
            table = pa.Table.from_batches(batches)

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
            ids = table.column(_id_name(alias))
            if missing == "insert":
                log.debug("ensuring all entities exist")
                self.add_entities(e_type, pc.unique(ids), duplicates="update")
            e_tbl = self._tables.get(e_type, None)
            if e_tbl is None:
                raise DataError(f"no entities of class {e_type}")

            e_ids = e_tbl.column(_id_name(e_type))
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
            link_nums[_num_name(alias)] = e_nums
            link_id_cols.add(_id_name(alias))

        new_table = pa.table(link_nums)
        if link_mask is not None:
            log.debug("filtering links to known entities")
            new_table = new_table.filter(link_mask)
        log.debug("adding %d new rows", new_table.num_rows)

        cur_table = self._tables[cls]
        if cur_table is not None:
            schema = pa.unify_schemas(
                [cur_table.schema, new_table.schema], promote_options="permissive"
            )
            batches = [b.cast(schema) for b in cur_table.to_batches()]
            batches += [b.cast(schema) for b in new_table.to_batches()]
            new_table = pa.Table.from_batches(batches)

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
        item_tbl = self._tables["item"]
        if item_tbl is not None:
            items = pd.Index(np.asarray(item_tbl.column("item_id")), name="item_id")
            items = Vocabulary(items, name="item")
        else:
            items = Vocabulary(name="item")

        user_tbl = self._tables.get("user", None)
        if user_tbl is not None:
            users = pd.Index(np.asarray(user_tbl.column("user_id")), name="user_id")
            users = Vocabulary(users, name="user")
        else:
            users = Vocabulary(name="user")

        idf = pd.DataFrame({"user_id": [], "item_id": []})
        for name, rc in self.schema.relationships.items():
            if not rc.interaction:
                continue

            tbl = self._tables[name]
            assert tbl is not None
            idf = tbl.to_pandas()
            for alias, e_type in rc.entities.items():
                e_type = e_type or alias
                e_tbl = self._tables[e_type]
                ids = pc.take(e_tbl.column(_id_name(e_type)), pa.array(idf[_num_name(alias)]))
                idf[_id_name(alias)] = np.asarray(ids)
                del idf[_num_name(alias)]

            break

        return MatrixDataset(users, items, idf)


def _id_name(name: str) -> str:
    return f"{name}_id"


def _num_name(name: str) -> str:
    return f"{name}_num"
