# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit dataset abstraction.
"""

# pyright: basic
from __future__ import annotations

import logging
import warnings
from typing import (
    Collection,
    Iterable,
    Literal,
    Optional,
    TypeAlias,
    TypeVar,
)

import numpy as np
import pandas as pd

from .builder import DatasetBuilder
from .dataset import Dataset
from .types import ID, AliasedColumn, Column, IDSequence
from .vocab import Vocabulary

DF_FORMAT: TypeAlias = Literal["numpy", "pandas", "torch"]
MAT_FORMAT: TypeAlias = Literal["scipy", "torch", "pandas", "structure"]
MAT_AGG: TypeAlias = Literal["count", "sum", "mean", "first", "last"]
LAYOUT: TypeAlias = Literal["csr", "coo"]
ACTION_FIELDS: TypeAlias = Literal["ratings", "timestamps"] | str

K = TypeVar("K")
_log = logging.getLogger(__name__)


USER_COLUMN = "user_id"
ITEM_COLUMN = "item_id"
USER_COMPAT_COLUMN = AliasedColumn(USER_COLUMN, ["user"], warn=True)
ITEM_COMPAT_COLUMN = AliasedColumn(ITEM_COLUMN, ["item"], warn=True)


def column_name(col: Column) -> str:
    match col:
        case str(name):
            return name
        case AliasedColumn(name=name):
            return name
        case _:  # pragma: nocover
            raise TypeError(f"invalid column spec {col}")


def normalize_columns(df: pd.DataFrame, *columns: Column) -> pd.DataFrame:
    """
    Resolve column aliases to columns, pulling them out of the index if necessary.

    Stability:
        Caller
    """

    for column in columns:
        name = column_name(column)
        if name in df.columns:
            continue
        elif name in df.index.names:
            df = df.reset_index(name)
            continue

        if not isinstance(column, AliasedColumn):
            raise KeyError("column %s not found", name)

        found = False
        for alias in column.compat_aliases:
            if alias in df.columns:
                found = True
                df = df.rename(columns={alias: name})
                if column.warn:
                    warnings.warn(f"found deprecated alias {alias} for {name}", DeprecationWarning)
                break
            elif alias in df.index.names:
                found = True
                df = df.reset_index(alias).rename(columns={alias: name})
                if column.warn:
                    warnings.warn(f"found deprecated alias {alias} for {name}", DeprecationWarning)
                break

        if not found:
            raise KeyError("column %s not found")

    return df


def from_interactions_df(
    df: pd.DataFrame,
    *,
    user_col: str | None = None,
    item_col: str | None = None,
    rating_col: str | None = None,
    timestamp_col: str | None = None,
    users: IDSequence | pd.Index | Iterable[ID] | Vocabulary | None = None,
    items: IDSequence | pd.Index | Iterable[ID] | Vocabulary | None = None,
) -> Dataset:
    """
    Create a dataset from a data frame of ratings or other user-item
    interactions.

    Stability:
        Caller
    Args:
        df:
            The user-item interactions (e.g. ratings).  The dataset code takes
            ownership of this data frame and may modify it.
        user_col:
            The name of the user ID column.  By default, looks for columns named
            ``user``, ``user_id``, or ``userId``, with several case variants.
        item_col:
            The name of the item ID column.  By default, looks for columns named
            ``item``, ``item_id``, or ``itemId``, with several case variants.
        rating_col:
            The name of the rating column.
        timestamp_col:
            The name of the timestamp column.
        user_ids:
            A vocabulary of user IDs.  The data frame is subset to this set of IDs.
        item_ids:
            A vocabulary of item IDs.  The data frame is subset to this set of IDs.
    Returns:
        The initiated data set.
    """
    _log.info("creating data set from %d x %d data frame", len(df.columns), len(df))
    df = normalize_interactions_df(
        df,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        timestamp_col=timestamp_col,
    )

    dsb = DatasetBuilder()

    if users is None and items is None:
        missing = "insert"
    else:
        missing = "filter"

        if users is None:
            users = df["user_id"].unique()
        else:
            users = np.asarray(users)
        dsb.add_entities("user", users)

        if items is None:
            items = df["item_id"].unique()
        else:
            items = np.asarray(items)
        dsb.add_entities("item", items)

    dsb.add_interactions(
        "rating", df, entities=["user", "item"], missing=missing, allow_repeats=False, default=True
    )

    return dsb.build()


def normalize_interactions_df(
    df: pd.DataFrame,
    *,
    user_col: Optional[str] = None,
    item_col: Optional[str] = None,
    rating_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Normalize the column names and layout for an interaction data frame.
    """
    _log.debug("normalizing data frame with columns %s", df.columns)
    if user_col is None:
        user_col = _find_column(
            df.columns,
            ["user_id", "user", "USER", "userId", "UserId"],
        )
    if user_col is None:  # pragma nocover
        raise ValueError("no user column found")
    if item_col is None:
        item_col = _find_column(
            df.columns,
            ["item_id", "item", "ITEM", "itemId", "ItemId"],
        )
    if item_col is None:  # pragma nocover
        raise ValueError("no item column found")
    if rating_col is None:
        rating_col = _find_column(
            df.columns,
            ["rating", "RATING"],
        )
    if timestamp_col is None:
        timestamp_col = _find_column(
            df.columns,
            ["timestamp", "TIMESTAMP"],
        )

    _log.debug("id columns: user=%s, item=%s", user_col, item_col)
    _log.debug("rating column: %s", rating_col)
    _log.debug("timestamp column: %s", timestamp_col)

    # rename and reorder columns
    known_columns = ["user_id", "item_id", "rating", "timestamp", "count"]
    renames = {user_col: "user_id", item_col: "item_id"}
    if rating_col:
        renames[rating_col] = "rating"
    if timestamp_col:
        renames[timestamp_col] = "timestamp"
    df = df.rename(columns=renames)
    kc = [c for c in known_columns if c in df.columns]
    oc = [c for c in df.columns if c not in known_columns]
    _log.debug("final columns: %s", kc + oc)
    return df[kc + oc]  # type: ignore


def _find_column(columns: Collection[str], acceptable: Iterable[str]) -> str | None:
    for col in acceptable:
        if col in columns:
            return col

    return None
