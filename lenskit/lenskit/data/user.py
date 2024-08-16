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

from .items import ItemList


class UserProfile:
    """
    Class representing a user's “profile” — their ID, history, and any other
    data known about them.  Data sources are encouraged to add additional fields
    to user profile objects based on specific data needs so that aware
    components can make use of them.

    Args:
        id:
            The ID of the user profile to create.  Can also be a
            ``UserProfile``, in which case the new profile is initialized as a
            shallow copy of the provided profile.  Other constructor arguments
            override the source profile in this case (so you can e.g. create a
            copy of a user with a different history).
        past_items:
            The user's item history, without duplicates.
    """

    id: EntityId | None = None
    """
    The user's identifier.

    .. note::
        This can be ``None`` to facilitate passing anonymous users and recommend
        based only on their history.
    """

    past_items: ItemList | None = None
    """
    The items the user has interacted with in the past.  Each item appears once,
    as in :meth:`lenskit.data.Dataset.interaction_matrix`.
    """

    def __init__(
        self, id: EntityId | UserProfile | None = None, *, past_items: ItemList | None = None
    ):
        if isinstance(id, UserProfile):
            self.id = id.id
            self.past_items = id.past_items
        else:
            self.id = id

        if past_items is not None:
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
