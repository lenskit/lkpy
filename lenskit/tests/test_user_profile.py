# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.data.items import ItemList
from lenskit.data.user import UserProfile


def test_empty_profile():
    user = UserProfile()
    assert user.id is None
    assert user.past_items is None


def test_profile_id():
    user = UserProfile("bob")
    assert user.id == "bob"
    assert user.past_items is None


def test_profile_history():
    items = ItemList(item_ids=[1, 7, 32])
    user = UserProfile(past_items=items)
    assert user.past_items is items


def test_profile_copy():
    items = ItemList(item_ids=[1, 7, 32])
    user = UserProfile("bob", past_items=items)
    assert user.id == "bob"
    assert user.past_items == items

    copy = UserProfile(user)
    assert copy.id == "bob"
    assert copy.past_items == items
    assert copy is not user
