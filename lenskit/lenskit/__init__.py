# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Toolkit for recommender systems research, teaching, and more.
"""

from importlib.metadata import PackageNotFoundError, version

from lenskit.algorithms import *  # noqa: F401,F403

try:
    __version__ = version("lenskit")
except PackageNotFoundError:
    # package is not installed
    __version__ = "UNKNOWN"
