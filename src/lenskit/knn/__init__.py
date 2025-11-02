# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
k-NN recommender models.
"""

from .item import ItemKNNConfig, ItemKNNScorer
from .slim import SLIMConfig, SLIMScorer
from .user import UserKNNConfig, UserKNNScorer

__all__ = [
    "ItemKNNScorer",
    "ItemKNNConfig",
    "UserKNNScorer",
    "UserKNNConfig",
    "SLIMScorer",
    "SLIMConfig",
]
