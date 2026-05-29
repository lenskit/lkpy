# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Legacy location of the MS Web import functions.

.. deprecated:: 2026.3

    Import from :mod:`lenskit.data` or :mod:`lenskit.data.sources.msweb` instead.
"""

import warnings

from .sources.msweb import load_ms_web

__all__ = ["load_ms_web"]

warnings.warn(
    "lenskit.data.msweb deprecated, use lenskit.data or lenskit.data.sources",
    DeprecationWarning,
)
