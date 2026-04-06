# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Data abstractions and data set access.
"""

from __future__ import annotations

from lenskit.diagnostics import FieldError

from ._adapt import from_interactions_df
from ._attributes import EntityAttribute
from ._batches import BatchedRange
from ._builder import DatasetBuilder
from ._dataset import Dataset, MatrixRelationshipSet, RelationshipSet
from ._entities import EntitySet
from ._items import ItemList
from ._query import QueryInput, QueryItemSource, RecQuery
from .amazon import load_amazon_ratings
from .collection import (
    GenericKey,
    ItemListCollection,
    ItemListCollector,
    ListILC,
    MutableItemListCollection,
    QueryIDKey,
    UserIDKey,
)
from .movielens import load_movielens, load_movielens_df
from .msweb import load_ms_web
from .types import ID, NPID, FeedbackType
from .vocab import Vocabulary

__all__ = [
    "Dataset",
    "EntitySet",
    "RelationshipSet",
    "MatrixRelationshipSet",
    "EntityAttribute",
    "DatasetBuilder",
    "FieldError",
    "from_interactions_df",
    "ID",
    "NPID",
    "FeedbackType",
    "ItemList",
    "ItemListCollection",
    "ItemListCollector",
    "MutableItemListCollection",
    "ListILC",
    "UserIDKey",
    "QueryIDKey",
    "GenericKey",
    "load_movielens",
    "load_movielens_df",
    "load_amazon_ratings",
    "load_ms_web",
    "Vocabulary",
    "RecQuery",
    "QueryInput",
    "QueryItemSource",
    "BatchedRange",
]
