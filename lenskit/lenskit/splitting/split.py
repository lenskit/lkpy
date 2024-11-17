# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import Literal, NamedTuple, TypeAlias

import pandas as pd

from lenskit.data import ID, Dataset, ItemList
from lenskit.data.bulk import dict_to_df

SplitTable: TypeAlias = Literal["matrix"]


class TTSplit(NamedTuple):
    """
    A train-test pair from splitting.
    """

    train: Dataset
    """
    The training data.
    """

    test: dict[ID, ItemList]
    """
    The test data.
    """

    @property
    def test_size(self) -> int:
        """
        Get the number of test pairs.
        """
        return sum(len(il) for il in self.test.values())

    @property
    def test_df(self) -> pd.DataFrame:
        """
        Get the test data as a data frame.
        """
        return dict_to_df(self.test)

    @property
    def train_df(self) -> pd.DataFrame:
        """
        Get the training data as a data frame.
        """
        return self.train.interaction_matrix("pandas", field="all")
