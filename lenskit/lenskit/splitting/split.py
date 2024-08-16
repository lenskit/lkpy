# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import Literal, NamedTuple, TypeAlias

import pandas as pd

from lenskit.data import Dataset, EntityId, ItemList

SplitTable: TypeAlias = Literal["matrix"]


class TTSplit(NamedTuple):
    """
    A train-test pair from splitting.
    """

    train: Dataset
    """
    The training data.
    """

    test: dict[EntityId, ItemList]
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


def dict_to_df(data: dict[EntityId, ItemList]) -> pd.DataFrame:
    """
    Convert a dictionary mapping user IDs to item lists into a data frame.
    """

    df = pd.concat(
        {u: il.to_df(numbers=False) for (u, il) in data.items()},
        names=["user_id"],
    )
    df = df.reset_index("user_id")
    df = df.reset_index(drop=True)
    return df


def dict_from_df(df: pd.DataFrame) -> dict[EntityId, ItemList]:
    """
    Convert a dictionary mapping user IDs to item lists into a data frame.
    """
    return {u: ItemList.from_df(udf) for (u, udf) in df.groupby("user_id")}  # type: ignore
