# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias, TypeVar

import pandas as pd

from lenskit.data import Dataset, ItemListCollection

SplitTable: TypeAlias = Literal["matrix"]
TK = TypeVar("TK", bound=tuple)


@dataclass
class TTSplit(Generic[TK]):
    """
    A train-test set from splitting or other sources.

    Stability:
        Caller
    """

    train: Dataset
    """
    The training data.
    """

    test: ItemListCollection[TK]
    """
    The test data.
    """

    name: str | None = None
    """
    A name for this train-test split.
    """

    @property
    def test_size(self) -> int:
        """
        Get the number of test pairs.
        """
        return sum(len(il) for il in self.test.lists())

    @property
    def test_df(self) -> pd.DataFrame:
        """
        Get the test data as a data frame.
        """
        return self.test.to_df()

    @property
    def train_df(self) -> pd.DataFrame:
        """
        Get the training data as a data frame.
        """
        return self.train.interaction_matrix(format="pandas", field="all")
