# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing_extensions import Literal, TypeAlias

from lenskit.types import EntityId, NPEntityId  # noqa: F401

FeedbackType: TypeAlias = Literal["explicit", "implicit"]
"Types of feedback supported."

from .dataset import Dataset, from_interactions_df  # noqa: F401, E402
from .items import ItemList  # noqa: F401, E402
from .movielens import load_movielens  # noqa: F401, E402
from .mtarray import MTArray, MTFloatArray, MTGenericArray, MTIntArray  # noqa: F401, E402
from .vocab import Vocabulary  # noqa: F401, E402
