"""
Definitions and utilities related to LensKit data schemas.
"""

import warnings

import pandas as pd

from .types import AliasedColumn, Column

USER_COLUMN = "user_id"
ITEM_COLUMN = "item_id"
USER_COMPAT_COLUMN = AliasedColumn(USER_COLUMN, ["user"], warn=True)
ITEM_COMPAT_COLUMN = AliasedColumn(ITEM_COLUMN, ["item"], warn=True)


def normalize_columns(df: pd.DataFrame, *columns: Column) -> pd.DataFrame:
    """
    Resolve column aliases to columns, pulling them out of the index if necessary.
    """

    for column in columns:
        name = column if isinstance(column, str) else column.name
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
