# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Legacy location of the MovieLens import functions.

.. deprecated:: 2026.3

    Import from :mod:`lenskit.data` or :mod:`lenskit.data.sources.movielens` instead.
"""

import warnings

from .sources.movielens import load_movielens, load_movielens_df

__all__ = ["load_movielens", "load_movielens_df"]

warnings.warn(
    "lenskit.data.movielens deprecated, use lenskit.data or lenskit.data.sources",
    DeprecationWarning,
)
