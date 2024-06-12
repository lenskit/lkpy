# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import Literal, TypeAlias

from .dataset import Dataset  # noqa: F401
from .matrix import RatingMatrix, sparse_ratings  # noqa: F401

FeedbackType: TypeAlias = Literal["explicit", "implicit"]
