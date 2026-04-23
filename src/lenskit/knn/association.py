# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Association-rule (conditional probability and lift) nearest-neighbor
recommendation.
"""

from typing import Literal

from pydantic import BaseModel, NonNegativeFloat

from lenskit.data import Dataset, ItemList, RecQuery
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

type AssociationMethod = Literal["probability", "lift"]


class AssociationConfig(BaseModel, extra="forbid"):
    """
    Configuration options for :class:`AssociationScorer`.
    """

    method: AssociationMethod = "probability"
    """
    The formula to use for computing item association level.
    """

    damping: NonNegativeFloat = 0.0
    r"""
    Damping factor (:math:`\kappa`) for `biased lift`_.

    .. _biased lift: https://md.ekstrandom.net/blog/2025/01/biased-lift
    """


class AssociationScorer(Component[ItemList], Trainable):
    r"""
    Item scorer using association rules to compute item relatedness.

    This scorer can compute item associations with three formulas:

    - Conditional probability (:math:`P[i|j]`), by setting
      :attr:`~AssociationConfig.method` to ``"probability"``.
    - Lift (:math:`\frac{P[i,j]}{P[i]P[j]}`), by setting
      :attr:`~AssociationConfig.method` to ``"lift"`` and
      :attr:`~AssociationConfig.damping` to 0.
    - `Biased lift`), by setting :attr:`~AssociationConfig.method` to ``"lift"``
      and :attr:`~AssociationConfig.damping` (:math:`\kappa`) to a positive
      value.

    .. _Biased lift: https://md.ekstrandom.net/blog/2025/01/biased-lift
    """

    config: AssociationConfig

    def train(self, data: Dataset, options: TrainingOptions):
        raise NotImplementedError()

    def __call__(self, query: RecQuery, items: ItemList):
        raise NotImplementedError()
