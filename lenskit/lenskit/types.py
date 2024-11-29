# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT
"""
Types used across LensKit.
"""

# pyright: strict
from __future__ import annotations

from typing import Any, Sequence, TypeAlias

import numpy as np

SeedLike: TypeAlias = int | np.integer[Any] | Sequence[int] | np.random.SeedSequence
"""
Type for RNG seeds (see `SPEC 7`_).

.. _SPEC 7: https://scientific-python.org/specs/spec-0007/
"""

RNGLike: TypeAlias = np.random.Generator | np.random.BitGenerator
"""
Type for random number generators as inputs (see `SPEC 7`_).

.. _SPEC 7: https://scientific-python.org/specs/spec-0007/
"""

RNGInput: TypeAlias = SeedLike | RNGLike | None
"""
Type for RNG inputs (see `SPEC 7`_).

.. _SPEC 7: https://scientific-python.org/specs/spec-0007/
"""
