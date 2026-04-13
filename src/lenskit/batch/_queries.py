# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import warnings
from collections.abc import Iterable, Iterator, Mapping, Sized
from typing import Any, Literal, NamedTuple, TypedDict

import pandas as pd

from lenskit.data import (
    ID,
    GenericKey,
    ItemList,
    ItemListCollection,
    QueryIDKey,
    RecQuery,
    UserIDKey,
    key_dict,
)
from lenskit.diagnostics import DataWarning


class BatchRecRequest(TypedDict, total=False):
    """
    Full recommendation request for evaluation, including candidate items.
    """

    query: RecQuery
    "Recommendation query."
    user_id: ID
    "User ID (ignored if :attr:`query` is specified)."
    query_id: ID
    "Query ID (ignored if :attr:`query` is specified, defaults to user ID)."
    items: ItemList
    "The items to score or possibly recommend."
    candidates: ItemList
    """
    Candidate items for the recommendations.  Overrides :attr:`items` for
    top-*N* recommendation.
    """
    test_items: ItemList
    """
    Test items for the recommendation query.  Overrides :attr:`items` for
    scoring or rating prediction.
    """


class BatchRequest(NamedTuple):
    """
    A single request for the batch inference runner.
    """

    query: RecQuery
    items: ItemList | None = None


class TestRequestAdapter(Iterable[BatchRecRequest], Sized):
    """
    Wrapper for an item list collection that interprets it as a collection of
    test requests.  Iterating over this collection will yield the requests in
    the same order they are in the underlying item list collection.

    The ``user_id`` and ``query_id`` key fields, if present are used to
    construct the recommendation queries.  The item lists themselves are
    interpreted as  directed by the ``items_as`` option:

    ``test``
        Items are used as test items (e.g., the items for which to predict
        ratings), but **not** candidates.

    ``candidates``
        Items are used as candidate lists.

    ``both``
        Items are used as both test items and candidates.

    ``None``
        Items are excluded.

    .. seealso::

        :ref:`batch-queries`, :class:`~lenskit.batch.BatchRecRequest`

    Args:
        lists:
            The item list collection.
        items_as:
            Where to put the item lists in the request.
    Warns:
        DataWarning:
            If the item list collection cannot be used to construct usable
            requests.
    """

    lists: ItemListCollection
    item_use: Literal["test", "candidates", "both"] | None

    def __init__(
        self,
        lists: ItemListCollection,
        *,
        items_as: Literal["test", "candidates", "both"] | None = "test",
    ):
        if "user_id" not in lists.key_fields:
            warnings.warn(
                "user_id is not in test data keys, requests unlikely to be usable",
                DataWarning,
                stacklevel=2,
            )
        self.lists = lists
        self.item_use = items_as

    def __len__(self):
        return len(self.lists)

    def __iter__(self) -> Iterator[BatchRecRequest]:
        for key, items in self.lists.items():
            kd = key_dict(key)
            req: BatchRecRequest = {}
            if uid := kd.get("user_id"):
                req["query_id"] = req["user_id"] = uid
            if qid := kd.get("query_id"):
                req["query_id"] = qid
            match self.item_use:
                case "test":
                    req["test_items"] = items
                case "candidates":
                    req["candidates"] = items
                case "both":
                    req["items"] = items
            yield req


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
