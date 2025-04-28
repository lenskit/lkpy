# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import warnings

from lenskit.logging import Stopwatch
from lenskit.random import derivable_rng, random_generator, set_global_rng

warnings.warn("lenskit.util is deprecated, import from original modules", DeprecationWarning)

__all__ = [
    "Stopwatch",
    "derivable_rng",
    "random_generator",
    "set_global_rng",
]
