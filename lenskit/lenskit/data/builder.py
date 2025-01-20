# pyright: strict
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import ArrayLike
from scipy.sparse import sparray

from .dataset import Dataset
from .schema import DataSchema, EntitySchema, RelationshipSchema
from .types import ID, NPID, CoreID

NPT = TypeVar("NPT", bound=np.generic)
NPArray1D: TypeAlias = np.ndarray[type[int], np.dtype[NPT]]

DuplicateAction: TypeAlias = Literal["update", "error", "overwrite"]
"""
Action to take on duplicate entities.
"""
MissingEntityAction: TypeAlias = Literal["add", "filter", "error"]
"""
Action to take when a relationship references a missing entity.
"""
IDArray: TypeAlias = Sequence[ID] | NPArray1D[NPID] | pa.IntegerArray[Any] | pa.StringArray


class DatasetBuilder:
    """
    Construct data sets from data and tables.
    """

    schema: DataSchema
    """
    The data schema assembled so far.  Do not modify this schema directly.
    """

    def entity_classes(self) -> dict[str, EntitySchema]:
        pass

    def relationship_classes(self) -> dict[str, RelationshipSchema]:
        pass

    def record_count(self, class_name: str) -> int:
        pass

    def add_entity_class(self, name: str) -> None:
        pass

    def add_relationship_class(
        self, name: str, *entities: str, duplicates_allowed: bool = True
    ) -> None:
        pass

    @overload
    def add_entities(
        self,
        cls: str,
        ids: IDArray | pd.Series[CoreID],
        /,
        *,
        duplicates: DuplicateAction = "error",
    ) -> None: ...
    @overload
    def add_entities(
        self, cls: str, frame: pa.Table | pd.DataFrame, /, *, duplicates: DuplicateAction = "error"
    ) -> None: ...
    def add_entities(
        self,
        cls: str,
        source: Sequence[ID]
        | NPArray1D[NPID]
        | pa.IntegerArray[Any]
        | pa.StringArray
        | pd.Series[CoreID]
        | pa.Table
        | pd.DataFrame,
        *,
        duplicates: DuplicateAction = "error",
    ) -> None:
        pass

    @overload
    def add_relationships(
        self,
        cls: str,
        data: pd.DataFrame | pa.Table,
        *,
        entities: Sequence[str] | None = None,
        missing: MissingEntityAction = "error",
    ) -> None: ...
    @overload
    def add_relationships(
        self,
        cls: str,
        *ids: IDArray,
        entities: Sequence[str] | None = None,
        missing: MissingEntityAction = "error",
    ) -> None: ...
    @overload
    def add_relationships(
        self,
        cls: str,
        entities: dict[str, IDArray],
        attributes: dict[str, ArrayLike],
        *,
        missing: MissingEntityAction = "error",
    ) -> None: ...
    def add_relationships(
        self,
        cls: str,
        *args: Any,
        missing: MissingEntityAction = "error",
        entities: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None: ...

    @overload
    def add_scalar_attribute(
        self, cls: str, name: str, data: pd.Series[Any] | pd.DataFrame | pa.Table, /
    ) -> None: ...
    @overload
    def add_scalar_attribute(
        self, cls: str, name: str, entities: IDArray | tuple[IDArray, ...], values: ArrayLike, /
    ) -> None: ...
    def add_scalar_attribute(
        self,
        cls: str,
        name: str,
        entities: IDArray | tuple[IDArray, ...] | pd.Series[Any] | pd.DataFrame | pa.Table,
        values: ArrayLike | None = None,
    ) -> None: ...

    @overload
    def add_list_attribute(
        self, cls: str, name: str, data: pd.Series[Any] | pd.DataFrame | pa.Table, /
    ) -> None: ...
    @overload
    def add_list_attribute(
        self, cls: str, name: str, entities: IDArray | tuple[IDArray, ...], values: ArrayLike, /
    ) -> None: ...
    def add_list_attribute(
        self,
        cls: str,
        name: str,
        entities: IDArray | tuple[IDArray, ...] | pd.Series[Any] | pd.DataFrame | pa.Table,
        values: ArrayLike | None = None,
    ) -> None: ...

    def add_vector_attribute(
        self,
        cls: str,
        name: str,
        entities: IDArray | tuple[IDArray, ...],
        values: ArrayLike | sparray,
        /,
        dims: ArrayLike | pd.Index[Any] | Sequence[Any] | None = None,
    ) -> None: ...

    def build(self) -> Dataset:
        raise NotImplementedError()
