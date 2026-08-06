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
from ._collection import (
    GenericKey,
    ItemListCollection,
    ItemListCollector,
    ListILC,
    MutableItemListCollection,
    QueryIDKey,
    UserIDKey,
    key_dict,
)
from ._container import DataContainer
from ._dataset import Dataset
from ._entities import EntitySet
from ._flatten import flatten_dict, unflatten_dict
from ._items import ItemList
from ._query import QueryInput, QueryItemSource, RecQuery
from ._relationships import MatrixRelationshipSet, RelationshipSet
from ._vocab import Vocabulary
from .msweb import load_ms_web
from .sources.amazon import load_amazon_ratings
from .sources.movielens import load_movielens, load_movielens_df
from .sources.steam import load_steam
from .types import ID, NPID, FeedbackType

__all__ = [
    "ID",
    "NPID",
    "BatchedRange",
    "DataContainer",
    "Dataset",
    "DatasetBuilder",
    "EntityAttribute",
    "EntitySet",
    "FeedbackType",
    "FieldError",
    "GenericKey",
    "ItemList",
    "ItemListCollection",
    "ItemListCollector",
    "ListILC",
    "MatrixRelationshipSet",
    "MutableItemListCollection",
    "QueryIDKey",
    "QueryInput",
    "QueryItemSource",
    "RecQuery",
    "RelationshipSet",
    "UserIDKey",
    "Vocabulary",
    "flatten_dict",
    "from_interactions_df",
    "key_dict",
    "load_amazon_ratings",
    "load_movielens",
    "load_movielens_df",
    "load_ms_web",
    "load_steam",
    "unflatten_dict",
]
