# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.data import ItemList, RecQuery, Vocabulary

ITEMS = ["a", "b", "c", "d", "e"]
VOCAB = Vocabulary(ITEMS)


def test_empty():
    query = RecQuery()
    assert query.user_id is None
    assert query.history_items is None
    assert query.user_items is None
    assert query.session_items is None
    assert query.context_items is None


def test_history_items():
    items = ItemList(ITEMS, vocabulary=VOCAB)
    query = RecQuery.create(items)
    assert query.user_id is None
    assert query.history_items == items
    assert query.user_items == items
    assert query.session_items is None
    assert query.context_items is None
    assert query.query_items == items


def test_session_items():
    items = ItemList(ITEMS, vocabulary=VOCAB)
    query = RecQuery(session_items=items)
    assert query.user_id is None
    assert query.history_items is None
    assert query.user_items is None
    assert query.session_items == items
    assert query.context_items is None
    assert query.query_items == items


def test_context_items():
    items = ItemList(ITEMS, vocabulary=VOCAB)
    query = RecQuery(context_items=items)
    assert query.user_id is None
    assert query.history_items is None
    assert query.user_items is None
    assert query.session_items is None
    assert query.context_items == items
    assert query.query_items == items


def test_context_items_override():
    c_il = ItemList(["a", "b"])
    s_il = ItemList(["c", "d"])
    h_il = ItemList(["e"])
    query = RecQuery(
        context_items=c_il,
        session_items=s_il,
        history_items=h_il,
    )
    assert query.user_id is None
    assert query.history_items == h_il
    assert query.user_items == h_il
    assert query.session_items == s_il
    assert query.context_items == c_il
    assert query.query_items == c_il


def test_session_items_override():
    s_il = ItemList(["c", "d"])
    h_il = ItemList(["e"])
    query = RecQuery(
        session_items=s_il,
        history_items=h_il,
    )
    assert query.user_id is None
    assert query.history_items == h_il
    assert query.user_items == h_il
    assert query.session_items == s_il
    assert query.context_items is None
    assert query.query_items == s_il


def test_combined_query_items():
    c_il = ItemList(["a", "b"])
    s_il = ItemList(["c", "d"])
    h_il = ItemList(["e"])
    query = RecQuery(
        context_items=c_il,
        session_items=s_il,
        history_items=h_il,
        query_source={"context", "history"},
    )
    assert query.user_id is None
    assert query.history_items == h_il
    assert query.user_items == h_il
    assert query.session_items == s_il
    assert query.context_items == c_il
    assert query.query_items is not None
    assert set(query.query_items.ids()) == {"a", "b", "e"}
