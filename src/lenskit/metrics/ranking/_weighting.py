# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Rank discounting models.
"""

from abc import ABC, abstractmethod
from typing import Annotated

import numpy as np
from annotated_types import Gt, Lt
from pydantic import NonNegativeInt, PositiveFloat, validate_call

from lenskit.data.types import NPVector


class RankWeight(ABC):
    """
    Interface for rank weighting models.  This returns *multiplicative* weights,
    so that scores should be multiplied by the weights in order to produce
    weighted scores.

    Stability:
        caller
    """

    @abstractmethod
    def weight(self, ranks: NPVector[np.int32]) -> NPVector[np.float64]:
        """
        Compute the discount for the specified ranks.

        Ranks must start with 1.
        """

    def log_weight(self, ranks: NPVector[np.int32]) -> NPVector[np.float64]:
        """
        Compute the (natural) log of the discount for the specified ranks.

        Ranks must start with 1.
        """
        return np.log(self.weight(ranks))

    def series_sum(self) -> float | None:
        """
        Get the sum of the infinite series of this discount function, if known.
        Some metrics (e.g. :func:`~lenskit.metrics.RBP`) will use this to
        normalize their measurements.
        """
        return None


class GeometricRankWeight(RankWeight):
    r"""
    Geometric cascade weighting for result ranks.  This is the ranking model
    used by RBP :citep:`rbp`.

    For patience :math:`p`, the discount is given by :math:`p^(k-1)`.  The
    sum of this infinite series is :math:`1 / (1 - p)`.

    Args:
        patience:
            The patience paramter :math:`p`.
    Stability:
        Caller
    """

    patience: float

    @validate_call
    def __init__(self, patience: Annotated[float, Gt(0.0), Lt(1.0)] = 0.85):
        self.patience = patience

    def weight(self, ranks) -> NPVector[np.float64]:
        return np.power(self.patience, ranks - 1)

    def log_weight(self, ranks) -> NPVector[np.float64]:
        return self.patience * np.log(ranks - 1)

    def series_sum(self) -> float:
        return 1 / (1 - self.patience)


class LogRankWeight(RankWeight):
    r"""
    Logarithmic weighting for result ranks.  This is the ranking model typically
    used for DCG and NDCG.

    Since :math:`\operatorname{lg} 1 = 0`, simply taking the log will result in
    division by 0 when weights are applied.  The correction for this in the
    original NDCG paper :cite:p:`ndcg` is to clip the ranks, so that both of the
    first two positions have discount :math:`\operatorname{lg} 2`.  A different
    correction somtimes seen is to compute :math:`\operatorname{lg} (k+1)`. This
    discount supports both; the default is to clip, but if the ``offset`` option
    is set to a positive number, it is added to the ranks instead.

    Args:
        base:
            The log base to use.
        offset:
            An offset to add to ranks before computing logs.
    """

    base: float
    offset: int

    @validate_call
    def __init__(self, *, base: PositiveFloat = 2, offset: NonNegativeInt = 0):
        self.base = base
        self.offset = offset

    def weight(self, ranks):
        if self.offset > 0:
            return np.reciprocal(np.log(ranks + self.offset) / np.log(self.base))
        else:
            return np.reciprocal(np.log(np.maximum(ranks, 2)) / np.log(self.base))
