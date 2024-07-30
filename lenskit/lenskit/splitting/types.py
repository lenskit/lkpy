# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import NamedTuple

import pandas as pd

from lenskit.data.dataset import Dataset


class TTPair(NamedTuple):
    """
    A train-test pair from splitting.
    """

    train: Dataset
    """
    The training data.
    """

    test: pd.DataFrame
    """
    The test data.
    """
