# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
EASE scoring model.
"""

from pydantic import BaseModel

from lenskit.data import Dataset, ItemList, RecQuery
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions


class EASEConfig(BaseModel):
    """
    Configuration for :class:`EASEScorer`.
    """

    regularization: float = 0.1
    """
    Regularization term for EASE.
    """


class EASEScorer(Component[ItemList], Trainable):
    """
    Score items using EASE :citep:`steckEmbarrassinglyShallowAutoencoders2019`.
    """

    config: EASEConfig
    """
    EASE configuration.
    """

    def train(self, data: Dataset, options: TrainingOptions):
        pass

    def __call__(self, query: RecQuery, items: ItemList):
        pass
