# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import warnings
from collections.abc import Iterable, Iterator, Mapping, Sized
from typing import Any, NamedTuple

import pandas as pd

from lenskit.data import (
    ID,
    GenericKey,
    ItemList,
    ItemListCollection,
    QueryIDKey,
    RecQuery,
    UserIDKey,
)


class BatchRequest(NamedTuple):
    """
    A single request for the batch inference runner.
    """

    query: RecQuery
    items: ItemList | None = None


def normalize_query_input(
    queries: Iterable[RecQuery]
    | Iterable[tuple[RecQuery, ItemList]]
    | Iterable[ID | GenericKey]
    | ItemListCollection[GenericKey]
    | Mapping[ID, ItemList]
    | pd.DataFrame,
) -> tuple[type[Any], Iterable[BatchRequest], int | None]:
    if isinstance(queries, pd.DataFrame):
        warnings.warn(
            "use an item list collection instead of a DataFrame (LKW-BATCHIN)",
            DeprecationWarning,
            stacklevel=2,
        )
        queries = ItemListCollection.from_df(queries)

    elif isinstance(queries, Mapping):
        warnings.warn(
            "query mappings are ambiguous and deprecated, use query lists (LKW-BATCHIN)",
            DeprecationWarning,
            stacklevel=2,
        )
        queries = ItemListCollection.from_dict(queries, "user_id")  # type: ignore

    if isinstance(queries, ItemListCollection):
        return queries.key_type, _ilc_queries(queries), len(queries)

    n = None
    if isinstance(queries, Sized):
        n = len(queries)

    q_iter = iter(queries)
    try:
        q_first = next(q_iter)
    except StopIteration:
        return tuple, [], 0

    fbr = _make_br(q_first)
    if fbr.query.query_id is not None:
        kt = QueryIDKey
    elif fbr.query.user_id is not None:
        kt = UserIDKey
    else:
        raise ValueError("query must have one of query_id, user_id")

    return kt, _iter_queries(q_first, q_iter), n


def _ilc_queries(queries: ItemListCollection):
    for q, items in queries.items():
        query = RecQuery(
            user_id=getattr(q, "user_id", None),
            query_id=getattr(q, "query_id", None),
        )
        yield BatchRequest(query, items)


def _iter_queries(
    first: RecQuery | tuple[RecQuery, ItemList] | ID | GenericKey,
    rest: Iterator[RecQuery | tuple[RecQuery, ItemList] | ID | GenericKey],
) -> Iterable[BatchRequest]:
    yield _make_br(first)
    for item in rest:
        yield _make_br(item)


def _make_br(q: RecQuery | tuple[RecQuery, ItemList] | ID | GenericKey) -> BatchRequest:
    if isinstance(q, RecQuery):
        return BatchRequest(q)
    elif isinstance(q, tuple):
        if isinstance(q[0], RecQuery):
            q, items = q
            return BatchRequest(q, items)  # type: ignore
        elif hasattr(q, "user_id"):
            # we have a named tuple with user IDs
            q = RecQuery(user_id=getattr(q, "user_id"))
            return BatchRequest(q)
        else:
            warnings.warn(
                "bare tuples are ambiguous and will be unsupported in 2026 (LKW-BATCHIN)",
                DeprecationWarning,
                stacklevel=3,
            )
            q = RecQuery(user_id=q)  # type: ignore
            return BatchRequest(q)
    else:
        q = RecQuery(user_id=q)
        return BatchRequest(q)
