"""
Functions for working with bulk per-user data.

.. note::

    This is a provisional API to enable incremental forward development while
    we develop more flexible abstractions for collections of data indexed by
    user or other keys.
"""

from typing import Iterator, cast, overload

import pandas as pd

from .items import ItemList
from .schemas import USER_COMPAT_COLUMN, column_name, normalize_columns
from .types import Column, EntityId


def dict_to_df(data: dict[EntityId, ItemList], *, column: str = "user_id") -> pd.DataFrame:
    """
    Convert a dictionary mapping user IDs to item lists into a data frame.

    Args:
        data:
            The dictionary of data.
        column:
            The column, to support dictionaries mapped by things other than user IDs.
    """

    df = pd.concat(
        {u: il.to_df(numbers=False) for (u, il) in data.items()},
        names=[column],
    )
    df = df.reset_index(column)
    df = df.reset_index(drop=True)
    return df


def dict_from_df(df: pd.DataFrame, *, column: str = "user_id") -> dict[EntityId, ItemList]:
    """
    Convert a dictionary mapping user IDs to item lists into a data frame.

    Args:
        df:
            The data frame.
        column:
            The column, to support dictionaries mapped by things other than user IDs.
    """
    return {u: ItemList.from_df(udf) for (u, udf) in df.groupby(column)}  # type: ignore


def group_df(df: pd.DataFrame, *, column: Column = USER_COMPAT_COLUMN):
    df = normalize_columns(df, column)
    col = column_name(column)
    return df.groupby(col)


@overload
def iter_groups(data: dict[EntityId, ItemList]) -> Iterator[tuple[EntityId, ItemList]]: ...
@overload
def iter_groups(
    data: pd.DataFrame | dict[EntityId, ItemList], *, column: Column = USER_COMPAT_COLUMN
) -> Iterator[tuple[EntityId, ItemList]]: ...
def iter_groups(
    data: pd.DataFrame | dict[EntityId, ItemList], *, column: Column = USER_COMPAT_COLUMN
) -> Iterator[tuple[EntityId, ItemList]]:
    if isinstance(data, pd.DataFrame):
        for key, df in group_df(data, column=column):
            yield cast(EntityId, key), ItemList.from_df(df)

    elif isinstance(data, dict):
        yield from data.items()

    else:  # pragma: nocover
        raise TypeError(f"unsupported data type {type(data)}")
