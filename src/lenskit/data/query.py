# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Recommendation queries.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from lenskit.data.items import ItemList

from .types import ID

QueryItemSource: TypeAlias = Literal["history", "session", "context"]


@dataclass
class RecQuery:
    """
    Representation of a the data available for a recommendation query.  This is
    generally the available inputs for a recommendation request *except* the set
    of candidate items.

    .. todo::

        Document and test methods for extending the recommendation query with arbitrary
        data to be used by client-provided pipeline components.

    .. todo::
        When LensKit supports context-aware recommendation, this should be extended
        to include context cues.

    Stability:
        Caller
    """

    user_id: ID | None = None
    """
    The user's identifier, if known.
    """

    history_items: ItemList | None = None
    """
    The items from the user's interaction history, with ratings if available.  This
    list is *deduplicated*, like :meth:`~lenskit.data.Dataset.interaction_matrix`,
    rather than a full interaction log.  As with :meth:`~lenskit.data.Dataset.user_row`,
    the rating is expected to be on a ``rating`` field, not ``score``.
    """

    session_items: ItemList | None = None
    """
    Items the user has interacted with in the current session.
    """

    context_items: ItemList | None = None
    """
    Items that form a context for the requested recommendations.  For example:

    -   To recommend products related to a product the user is viewing, the
        viewed product is a context item.
    -   To recommend products to add to a shopping cart, the current cart
        contents are the context items.
    """

    query_source: QueryItemSource | set[QueryItemSource] | None = None
    """
    The list of items to return from :meth:`query_items`.
    """

    user_items: ItemList | None = None
    """
    Deprecated alias for :attr:`history_items`.

    .. deprecated:: 2025.6

        Use :attr:`history_items` instead.  This property will be removed in
        LensKit 2026.1.
    """

    def __post_init__(self):
        if self.user_items is None:
            self.user_items = self.history_items
        elif self.history_items is None:
            warnings.warn("user_items is deprecated, use history_items", DeprecationWarning, 2)
            self.history_items = self.user_items

    @classmethod
    def create(cls, data: QueryInput) -> RecQuery:
        """
        Create a recommendation query from an input, filling in available
        components based on the data type.

        Args:
            data:
                Input data to turn into a recommendation query.  If the input is
                already a query, it is returned *as-is* (not copied).

        Returns:
            The recommendation query.
        """
        if data is None:
            return RecQuery()
        elif isinstance(data, RecQuery):
            return data
        elif isinstance(data, ItemList):
            return cls(history_items=data)
        elif isinstance(data, np.integer):
            return cls(user_id=data.item())
        elif isinstance(data, int | str | bytes):
            return cls(user_id=data)
        else:  # pragma: nocover
            raise TypeError(f"invalid query input (type {type(data)})")

    @property
    def query_items(self) -> ItemList | None:
        """
        Return the items the recommender should treat as a “query”.

        This property exists so that components can obtain a list of items to
        use for generating recommendations, regardless of the precise shape of
        recommendation problem being solved (e.g., user-personalized
        recommendation or session-based recommendation).  This allows general
        models such as :class:`~lenskit.knn.ItemKNNScorer` to operate across
        recommendation tasks.

        The :attr:`query_source` attribute determines which list(s) of items
        in this query are considered the “query items”.  If unset, the first
        of the following that is available are used as the query items:

        1. :attr:`context_items`
        2. :attr:`session_items`
        3. :attr:`history_items`
        """
        if self.query_source is None:
            if self.context_items is not None:
                return self.context_items
            if self.session_items is not None:
                return self.session_items
            if self.history_items is not None:
                return self.history_items
        elif isinstance(self.query_source, str):
            return self._items(self.query_source)
        else:
            items = None
            for src in self.query_source:
                il2 = self._items(src)
                if il2 is not None:
                    if items is None:
                        items = il2
                    else:
                        items = items.concat(il2)

            return items

    def _items(self, source: QueryItemSource | None):
        match source:
            case "context":
                return self.context_items
            case "session":
                return self.session_items
            case "history":
                return self.history_items
            case _:  # pragma: nocover
                raise ValueError(f"unsupported item source {source}")


QueryInput: TypeAlias = RecQuery | ID | ItemList | None
"""
Types that can be converted to a query by :meth:`RecQuery.create`.
"""
