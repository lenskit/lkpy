# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Splitting data for train-test evaluation.
"""

import numpy as np

from lenskit.data import Dataset

from .holdout import HoldoutMethod, LastFrac, LastN, SampleFrac, SampleN
from .records import crossfold_records, sample_records
from .split import TTSplit
from .users import crossfold_users, sample_users

__all__ = [
    "TTSplit",
    "HoldoutMethod",
    "LastFrac",
    "LastN",
    "SampleFrac",
    "SampleN",
    "crossfold_records",
    "sample_records",
    "crossfold_users",
    "sample_users",
    "simple_test_pair",
]


def simple_test_pair(
    ratings: Dataset, n_users=200, n_rates=5, f_rates=None, rng: np.random.Generator | None = None
) -> TTSplit:
    """
    Return a single, basic train-test pair for some ratings.  This is only intended
    for convenience use in test and demos - do not use for research.
    """

    if f_rates:
        samp = SampleFrac(f_rates, rng=rng)
    else:
        samp = SampleN(n_rates, rng=rng)

    return sample_users(ratings, n_users, samp, rng=rng)
