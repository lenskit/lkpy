"""
Recommendation queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from lenskit.data.items import ItemList

from .types import ID


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
    """

    user_id: ID | None = None
    """
    The user's identifier.
    """
    user_items: ItemList | None = None
    """
    The items from the user's interaction history, with ratings if available.  This
    list is *deduplicated*, like :meth:`~lenskit.data.Dataset.interaction_matrix`,
    rather than a full interaction log.  As with :meth:`~lenskit.data.Dataset.user_row`,
    the rating is expected to be on a ``rating`` field, not ``score``.
    """

    @classmethod
    def create(cls, data: QueryInput) -> RecQuery:
        """
        Create a recommenadtion query from an input, filling in available
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
            return cls(user_items=data)
        elif isinstance(data, np.integer):
            return cls(user_id=data.item())
        elif isinstance(data, int | str | bytes):
            return cls(user_id=data)
        else:  # pragma: nocover
            raise TypeError(f"invalid query input (type {type(data)})")


QueryInput: TypeAlias = RecQuery | ID | ItemList | None
"""
Types that can be converted to a query by :meth:`RecQuery.create`.
"""
