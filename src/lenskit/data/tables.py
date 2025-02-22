# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class TorchUserItemTable:
    """
    Table of user-item interaction data represented as PyTorch tensors.

    Stability:
        Caller
    """

    user_nums: torch.Tensor
    """
    User numbers (0-based contiguous integers, see :ref:`data-identifiers`).
    """
    item_nums: torch.Tensor
    """
    Item numbers (0-based contiguous integers, see :ref:`data-identifiers`).
    """
    ratings: Optional[torch.Tensor] = None
    """
    Ratings for the items.
    """
    timestamps: Optional[torch.Tensor] = None
    """
    Timestamps for recorded user-item interactions.
    """


@dataclass
class NumpyUserItemTable:
    """
    Table of user-item interaction data represented as NumPy arrays.

    Stability:
        Caller
    """

    user_nums: np.ndarray[int, np.dtype[np.int32]]
    """
    User numbers (0-based contiguous integers, see :ref:`data-identifiers`).
    """
    item_nums: np.ndarray[int, np.dtype[np.int32]]
    """
    Item numbers (0-based contiguous integers, see :ref:`data-identifiers`).
    """
    ratings: Optional[np.ndarray[int, np.dtype[np.float32]]] = None
    """
    Ratings for the items.
    """
    timestamps: Optional[np.ndarray[int, np.dtype[np.int64]]] = None
    """
    Timestamps for recorded user-item interactions.
    """
