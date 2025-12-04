# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Data abstractions and data set access.
"""

from __future__ import annotations

from lenskit.diagnostics import FieldError

from .adapt import from_interactions_df
from .amazon import load_amazon_ratings
from .attributes import AttributeSet
from .batches import BatchIter
from .builder import DatasetBuilder
from .collection import (
    GenericKey,
    ItemListCollection,
    ItemListCollector,
    ListILC,
    MutableItemListCollection,
    QueryIDKey,
    UserIDKey,
)
from .dataset import Dataset, EntitySet, MatrixRelationshipSet, RelationshipSet
from .items import ItemList
from .matrix import COOStructure, CSRStructure
from .movielens import load_movielens, load_movielens_df
from .msweb import load_ms_web
from .mtarray import MTArray, MTFloatArray, MTGenericArray, MTIntArray
from .query import QueryInput, RecQuery
from .types import ID, NPID, FeedbackType
from .vocab import Vocabulary

__all__ = [
    "Dataset",
    "EntitySet",
    "RelationshipSet",
    "MatrixRelationshipSet",
    "AttributeSet",
    "DatasetBuilder",
    "FieldError",
    "CSRStructure",
    "COOStructure",
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
    "MTArray",
    "MTFloatArray",
    "MTGenericArray",
    "MTIntArray",
    "Vocabulary",
    "RecQuery",
    "QueryInput",
    "BatchIter",
]
