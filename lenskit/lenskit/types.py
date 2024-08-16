# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

from typing import TypeAlias

import numpy as np
from seedbank import SeedLike

RandomSeed: TypeAlias = SeedLike | np.random.Generator | np.random.RandomState
