# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
k-NN recommender models.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["item", "user", "ease"],
    submod_attrs={
        "item": ["ItemKNNScorer", "ItemKNNConfig"],
        "user": ["UserKNNScorer", "UserKNNConfig"],
        "ease": ["EASEScorer", "EASEConfig"],
    },
)
