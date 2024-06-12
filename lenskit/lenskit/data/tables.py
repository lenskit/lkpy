from __future__ import annotations

import numpy as np
import torch
from typing_extensions import (
    NamedTuple,
    Optional,
)


class TorchUserItemTable(NamedTuple):
    """
    Table of user-item interaction data represented as PyTorch tensors.
    """

    user_nums: torch.Tensor
    """
    User numbers (0-based contiguous integers, see :ref:`data-identifiers`).
    """
    item_nums: torch.Tensor
    """
    Item numbers (0-based contiguous integers, see :ref:`data-identifiers`).
    """
    ratings: Optional[torch.Tensor]
    """
    Ratings for the items.
    """
    timestamps: Optional[torch.Tensor]
    """
    Timestamps for recorded user-item interactions.
    """


class NumpyUserItemTable(NamedTuple):
    """
    Table of user-item interaction data represented as NumPy arrays.
    """

    user_nums: np.ndarray[int, np.dtype[np.int32]]
    """
    User numbers (0-based contiguous integers, see :ref:`data-identifiers`).
    """
    item_nums: np.ndarray[int, np.dtype[np.int32]]
    """
    Item numbers (0-based contiguous integers, see :ref:`data-identifiers`).
    """
    ratings: Optional[np.ndarray[int, np.dtype[np.float32]]]
    """
    Ratings for the items.
    """
    timestamps: Optional[np.ndarray[int, np.dtype[np.int64]]]
    """
    Timestamps for recorded user-item interactions.
    """
