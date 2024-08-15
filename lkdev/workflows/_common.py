# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import Any

from lkdev.ghactions import GHStep

PACKAGES = ["lenskit", "lenskit-funksvd", "lenskit-implicit", "lenskit-hpf"]


def step_checkout(options: Any = None, depth: int = 0) -> GHStep:
    return {
        "name": "🛒 Checkout",
        "uses": "actions/checkout@v4",
        "with": {"fetch-depth": depth},
    }
