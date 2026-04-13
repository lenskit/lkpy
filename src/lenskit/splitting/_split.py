# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, TypeAlias

import pandas as pd

from lenskit.batch import TestRequestAdapter
from lenskit.data import Dataset, ItemListCollection
from lenskit.diagnostics import DataWarning

SplitTable: TypeAlias = Literal["matrix"]


@dataclass
class TTSplit:
    """
    A train-test set from splitting or other sources.

    .. versionchanged:: 2026.1

        Added the :attr:`test_requests` property.

    Stability:
        Caller
    """

    train: Dataset
    """
    The training data.
    """

    test: ItemListCollection
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

    @property
    def test_requests(self) -> TestRequestAdapter:
        """
        Get the test data as a sequence of batch recommendation requests.

        .. seealso::

            :ref:`batch-queries`

        Returns:
            An collection that iterates over recommendation requests derived
            from the test data.
        """
        if "user_id" not in self.test.key_fields:
            warnings.warn(
                "user_id is not in test data keys, requests unlikely to be usable",
                DataWarning,
                stacklevel=2,
            )
        return TestRequestAdapter(self.test)
