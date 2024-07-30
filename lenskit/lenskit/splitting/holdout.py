# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Per-user rating holdout methods for user-based data splitting.
"""

from typing import Protocol

from seedbank import numpy_rng


class HoldoutMethod(Protocol):
    """
    Holdout methods select test rows for a user (or item).  Partition methods
    are callable; when called with a data frame, they return the test entries.
    """

    def __call__(self, udf):
        """
        Subset a data frame.

        Args:
            udf(pandas.DataFrame):
                The input data frame of rows for a user or item.

        Returns:
            pandas.DataFrame:
                The data frame of test rows, a subset of ``udf``.
        """
        pass


class SampleN(HoldoutMethod):
    """
    Randomly select a fixed number of test rows per user/item.

    Args:
        n(int): the number of test items to select
        rng: the random number generator or seed
    """

    def __init__(self, n, rng_spec=None):
        self.n = n
        self.rng = numpy_rng(rng_spec)

    def __call__(self, udf):
        return udf.sample(n=self.n, random_state=self.rng)


class SampleFrac(HoldoutMethod):
    """
    Randomly select a fraction of test rows per user/item.

    Args:
        frac(float): the fraction items to select for testing.
    """

    def __init__(self, frac, rng_spec=None):
        self.fraction = frac
        self.rng = numpy_rng(rng_spec)

    def __call__(self, udf):
        return udf.sample(frac=self.fraction, random_state=self.rng)


class LastN(HoldoutMethod):
    """
    Select a fixed number of test rows per user/item, based on ordering by a
    column.

    Args:
        n(int): The number of test items to select.
    """

    def __init__(self, n, col="timestamp"):
        self.n = n
        self.column = col

    def __call__(self, udf):
        return udf.sort_values(self.column).iloc[-self.n :]


class LastFrac(HoldoutMethod):
    """
    Select a fraction of test rows per user/item.

    Args:
        frac(double): the fraction of items to select for testing.
    """

    def __init__(self, frac, col="timestamp"):
        self.fraction = frac
        self.column = col

    def __call__(self, udf):
        n = round(len(udf) * self.fraction)
        return udf.sort_values(self.column).iloc[-n:]
