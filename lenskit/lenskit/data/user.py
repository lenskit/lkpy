# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Basic user information wrapper.
"""

from __future__ import annotations

from typing_extensions import override

from lenskit.types import EntityId

from .items import HasItemList, ItemList


class UserProfile(HasItemList):
    """
    Class representing a user's “profile” — their ID, history, and any other
    data known about them.  Data sources are encouraged to add additional fields
    to user profile objects based on specific data needs, but components will
    not know about them.
    """

    id: EntityId | None
    """
    The user's identifier.

    .. note::
        This can be ``None`` to facilitate passing anonymous users and recommend
        based only on their history.
    """

    past_items: ItemList | None
    """
    The items the user has interacted with in the past.  Each item appears once,
    as in :meth:`lenskit.data.Dataset.interaction_matrix`.
    """

    def __init__(self, id: EntityId | None = None, *, past_items: ItemList | None = None):
        self.id = id
        self.past_items = past_items

    @override
    def item_list(self) -> ItemList | None:
        """
        Return the item list.
        """
        return self.past_items

    def __str__(self) -> str:
        s = "<"
        if self.id is not None:
            s += f"User “{self.id}”"
        else:
            s += "AnonymousUser"
        if self.past_items is not None:
            s += f" with {len(self.past_items)} items"
        s += ">"
        return s

    def __repr__(self) -> str:
        return str(self)
