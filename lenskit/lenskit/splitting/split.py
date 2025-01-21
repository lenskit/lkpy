# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias, TypeVar

import pandas as pd
import pyarrow as pa

from lenskit.data import Dataset, DatasetBuilder, ItemListCollection

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

    @classmethod
    def from_src_and_test(cls, src: Dataset, test: ItemListCollection[TK]) -> TTSplit[TK]:
        """
        Create a split by subtracting test data from a source dataset.
        """
        test_df = test.to_df()

        iname = src.default_interaction_class()
        train_build = DatasetBuilder(src)
        train_build.filter_interactions(
            iname,
            remove=pa.table(
                {
                    "user_num": pa.array(src.users.numbers(test_df["user_id"])),
                    "item_num": pa.array(src.items.numbers(test_df["item_id"])),
                }
            ),
        )

        train = train_build.build()

        return cls(train, test)
