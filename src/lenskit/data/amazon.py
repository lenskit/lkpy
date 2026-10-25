# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Legacy location of the Amazon import functions.

.. deprecated:: 2026.3

    Import from :mod:`lenskit.data` or :mod:`lenskit.data.sources.movielens` instead.
"""

import warnings

from .sources.amazon import load_amazon_ratings

__all__ = ["load_amazon_ratings"]

warnings.warn(
    "lenskit.data.amazon deprecated, use lenskit.data or lenskit.data.sources", DeprecationWarning
)
