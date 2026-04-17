# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Recommendation queries.
"""

# pyright: strict
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np

from ._items import ItemList
from .types import ID

type QueryItemSource = Literal["history", "session", "context"]
"""
Valid sources for query items.
"""

type QueryInput = RecQuery | ID | ItemList | None
"""
Types that can be converted to a query by :meth:`RecQuery.create`.
"""


@dataclass(kw_only=True)
class RecQuery:
    """
    Representation of a the data available for a recommendation query.  This is
    generally the available inputs for a recommendation request *except* the set
    of candidate items.

    .. versionchanged:: 2026.1

        Made arguments keyword-only, and removed the historical ``user_items`` attribute.

    .. todo::

        Document and test methods for extending the recommendation query with arbitrary
        data to be used by client-provided pipeline components.

    .. todo::
        When LensKit supports context-aware recommendation, this should be extended
        to include context cues.

    Stability:
        Caller
    """

    query_id: ID | tuple[ID, ...] | None = None
    """
    An identifier for this query.

    Query identifiers are used for things like mapping batch recommendation
    outputs to their inputs.
    """

    query_time: datetime | None = None
    """
    The time at which the query is issued.

    .. note::

        No LensKit models or data processing code currently makes use of this,
        but it is included for to support future time-aware modeling and replays
        of historical data.
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
            return self.combined_items(*self.query_source)

    @property
    def all_items(self) -> ItemList | None:
        """
        Get a single list of all items mentioned in the query.
        """
        return self.combined_items("history", "session", "context")

    def combined_items(self, *sources: QueryItemSource) -> ItemList | None:
        """
        Obtain a combined list of items from one or more sources.
        """
        items = None
        for src in sources:
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
