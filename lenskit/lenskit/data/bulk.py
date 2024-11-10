"""
Functions for working with bulk per-user data.

.. note::

    This is a provisional API to enable incremental forward development while
    we develop more flexible abstractions for collections of data indexed by
    user or other keys.
"""

import pandas as pd

from .items import ItemList
from .types import EntityId


def dict_to_df(data: dict[EntityId, ItemList], column: str = "user_id") -> pd.DataFrame:
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


def dict_from_df(df: pd.DataFrame, column: str = "user_id") -> dict[EntityId, ItemList]:
    """
    Convert a dictionary mapping user IDs to item lists into a data frame.

    Args:
        df:
            The data frame.
        column:
            The column, to support dictionaries mapped by things other than user IDs.
    """
    return {u: ItemList.from_df(udf) for (u, udf) in df.groupby(column)}  # type: ignore
