# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT
"""
Data abstractions and data set access.
"""

from __future__ import annotations

from .collection import GenericKey, ItemListCollection, UserIDKey
from .convert import from_interactions_df
from .dataset import Dataset, FieldError
from .items import ItemList
from .movielens import load_movielens, load_movielens_df
from .mtarray import MTArray, MTFloatArray, MTGenericArray, MTIntArray
from .query import QueryInput, RecQuery
from .types import ID, NPID, FeedbackType, UITuple
from .vocab import Vocabulary

__all__ = [
    "Dataset",
    "FieldError",
    "from_interactions_df",
    "ID",
    "NPID",
    "UITuple",
    "FeedbackType",
    "ItemList",
    "ItemListCollection",
    "UserIDKey",
    "GenericKey",
    "load_movielens",
    "load_movielens_df",
    "MTArray",
    "MTFloatArray",
    "MTGenericArray",
    "MTIntArray",
    "Vocabulary",
    "RecQuery",
    "QueryInput",
]
