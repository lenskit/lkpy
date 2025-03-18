# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Flexible PyTorch matrix factorization models for LensKit.

The components in this package implement several matrix factorization models for
LensKit, and also serve as an example for practical PyTorch recommender
training.

.. stability:: internal
"""

from ._base import FlexMFConfigBase, FlexMFScorerBase
from ._explicit import FlexMFExplicitConfig, FlexMFExplicitScorer

__all__ = [
    "FlexMFConfigBase",
    "FlexMFScorerBase",
    "FlexMFExplicitConfig",
    "FlexMFExplicitScorer",
]
