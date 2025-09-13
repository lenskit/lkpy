# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Graph-based models, especially GNNs with :mod:`torch_geometric`.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules="lightgcn",
    submod_attrs={
        "lightgcn": ["LightGCNScorer", "LightGCNConfig"],
    },
)
