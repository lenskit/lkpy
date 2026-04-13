# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Mapping, Sized
from dataclasses import dataclass
from typing import Literal, TypedDict

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

type BatchInput = (
    Iterable[BatchRecRequest]
    | Iterable[RecQuery]
    | Iterable[ID | GenericKey]
    | ItemListCollection[GenericKey]
)
"""
Allowed input types for batch inference routines.
"""


class BatchRecRequest(TypedDict, total=False):
    """
    Full recommendation request for batch inference, including candidate items.

    Stability:
        Full
    """

    query: RecQuery
    "Recommendation query."
    user_id: ID
    "User ID (ignored if :attr:`query` is specified)."
    query_id: ID | GenericKey
    "Query ID (ignored if :attr:`query` is specified, defaults to user ID)."
    items: ItemList
    """
    The items to score or possibly recommend. It is usually better to supply
    :attr:`candidates` and/or :attr:`test_items`.
    """
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


@dataclass
class ResolvedBatchRequest:
    """
    A single request for the batch inference runner, with fully-resolved
    defaults.

    Stability:
        Internal
    """

    query: RecQuery
    test_items: ItemList | None = None
    candidates: ItemList | None = None


class TestRequestAdapter(Iterable[BatchRecRequest], Sized):
    """
    Wrapper for an item list collection that interprets it as a collection of
    test requests.  Iterating over this collection will yield the requests in
    the same order they are in the underlying item list collection.

    The ``user_id`` and ``query_id`` key fields, if present are used to
    construct the recommendation queries.  If there is no ``query_id`` field,
    then the entire key is used as a query ID.  The item lists themselves are
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
    Stability:
        Caller
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
                req["user_id"] = uid
            if qid := kd.get("query_id"):
                req["query_id"] = qid
            else:
                req["query_id"] = key
            match self.item_use:
                case "test":
                    req["test_items"] = items
                case "candidates":
                    req["candidates"] = items
                case "both":
                    req["items"] = items
            yield req


def normalize_query_input(
    queries: BatchInput,
) -> tuple[type[GenericKey], Iterable[ResolvedBatchRequest], int | None]:
    kt = None

    if isinstance(queries, ItemListCollection):
        kt = queries.key_type
        queries = TestRequestAdapter(queries)
    elif isinstance(queries, Mapping):
        raise TypeError("mappings are no longer a supported batch input")
    elif isinstance(queries, pd.DataFrame):
        raise TypeError("data frames are no longer a supported batch input")

    n = None
    if isinstance(queries, Sized):
        n = len(queries)

    q_iter = iter(queries)
    try:
        q_first = next(q_iter)
    except StopIteration:
        return tuple, [], 0

    q_first = _resolve_batch_request(q_first)
    if isinstance(q_first.query.query_id, tuple):
        kt = type(q_first.query.query_id)
    elif q_first.query.query_id is not None:
        kt = QueryIDKey
    elif q_first.query.user_id is not None:
        kt = UserIDKey
    else:
        raise ValueError("query must have one of query_id, user_id")

    return kt, _iter_queries(q_first, q_iter), n


def _iter_queries(
    first: ResolvedBatchRequest,
    rest: Iterable[BatchRecRequest] | Iterable[RecQuery] | Iterable[ID | GenericKey],
) -> Iterable[ResolvedBatchRequest]:
    yield first
    for item in rest:
        yield _resolve_batch_request(item)


def _resolve_batch_request(q: RecQuery | ID | GenericKey | BatchRecRequest) -> ResolvedBatchRequest:
    if isinstance(q, RecQuery):
        return ResolvedBatchRequest(query=q)
    elif isinstance(q, tuple):
        user_id = getattr(q, "user_id", None)
        query_id = getattr(q, "query_id", q)
        return ResolvedBatchRequest(RecQuery(user_id=user_id, query_id=query_id))
    elif isinstance(q, Mapping):
        query = q.get("query", None)
        if query is None:
            query = RecQuery(user_id=q.get("user_id"), query_id=q.get("query_id"))

        rq = ResolvedBatchRequest(query=query)
        if (items := q.get("candidates")) is not None:
            rq.candidates = items
        if (items := q.get("test_items")) is not None:
            rq.test_items = items
        if (items := q.get("items")) is not None:
            if rq.candidates is None:
                rq.candidates = items
            if rq.test_items is None:
                rq.test_items = items
        return rq
    else:
        return ResolvedBatchRequest(RecQuery(user_id=q))
