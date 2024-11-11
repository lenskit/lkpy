"""
Functions for working with bulk per-user data.

.. note::

    This is a provisional API to enable incremental forward development while
    we develop more flexible abstractions for collections of data indexed by
    user or other keys.
"""

from collections.abc import Mapping
from typing import Iterator, cast, overload

import pandas as pd

from .items import ItemList
from .schemas import (
    ITEM_COMPAT_COLUMN,
    USER_COLUMN,
    USER_COMPAT_COLUMN,
    column_name,
    normalize_columns,
)
from .types import Column, EntityId


def dict_to_df(
    data: Mapping[EntityId, ItemList | None], *, column: str = USER_COLUMN
) -> pd.DataFrame:
    """
    Convert a dictionary mapping user IDs to item lists into a data frame.
    Missing item lists are excluded.

    Args:
        data:
            The dictionary of data.
        column:
            The column, to support dictionaries mapped by things other than user
            IDs.
    """

    df = pd.concat(
        {u: il.to_df(numbers=False) for (u, il) in data.items() if il is not None},
        names=[column],
    )
    df = df.reset_index(column)
    df = df.reset_index(drop=True)
    return df


def dict_from_df(
    df: pd.DataFrame, *, column: Column = USER_COMPAT_COLUMN, item_col: Column = ITEM_COMPAT_COLUMN
) -> dict[EntityId, ItemList]:
    """
    Convert a dictionary mapping user IDs to item lists into a data frame.

    Args:
        df:
            The data frame.
        column:
            The column, to support dictionaries mapped by things other than user IDs.
    """
    df = normalize_columns(df, column, item_col)
    return {u: ItemList.from_df(udf) for (u, udf) in df.groupby(column_name(column))}  # type: ignore


def group_df(df: pd.DataFrame, *, column: Column = USER_COMPAT_COLUMN, item_col=ITEM_COMPAT_COLUMN):
    """
    Group a data frame by a specified column, possibly checking for and
    normalizing the names of other columns as well.  The default options group
    by ``user_id`` and require an ``item_id`` column, allowing the compatibility
    aliases ``user`` and ``item``.
    """
    df = normalize_columns(df, column, item_col)
    col = column_name(column)
    return df.groupby(col)


@overload
def count_item_lists(data: Mapping[EntityId, ItemList | None]) -> int: ...
@overload
def count_item_lists(
    data: pd.DataFrame | Mapping[EntityId, ItemList | None],
    *,
    column: Column = USER_COMPAT_COLUMN,
) -> int: ...
def count_item_lists(
    data: pd.DataFrame | Mapping[EntityId, ItemList | None],
    *,
    column: Column = USER_COMPAT_COLUMN,
) -> int:
    if isinstance(data, pd.DataFrame):
        data = normalize_columns(data, column)
        return data[column_name(column)].nunique()
    else:
        return len(data)


@overload
def iter_item_lists(
    data: Mapping[EntityId, ItemList | None],
) -> Iterator[tuple[EntityId, ItemList | None]]: ...
@overload
def iter_item_lists(
    data: pd.DataFrame | Mapping[EntityId, ItemList | None],
    *,
    column: Column = USER_COMPAT_COLUMN,
    item_col=ITEM_COMPAT_COLUMN,
) -> Iterator[tuple[EntityId, ItemList | None]]: ...
def iter_item_lists(
    data: pd.DataFrame | Mapping[EntityId, ItemList | None],
    *,
    column: Column = USER_COMPAT_COLUMN,
    item_col=ITEM_COMPAT_COLUMN,
) -> Iterator[tuple[EntityId, ItemList | None]]:
    """
    Iterate over item lists identified by keys.  When the input is a data frame,
    the column names may be specified; the default options group by ``user_id``
    and require an ``item_id`` column, allowing the compatibility aliases
    ``user`` and ``item``.
    """
    if isinstance(data, pd.DataFrame):
        for key, df in group_df(data, column=column, item_col=item_col):
            yield cast(EntityId, key), ItemList.from_df(df)

    else:
        yield from data.items()
