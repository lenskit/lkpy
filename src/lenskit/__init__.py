# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Recommender systems toolkit.
"""

from importlib.metadata import PackageNotFoundError, version

import lazy_loader as lazy

try:
    __version__ = version("lenskit")
except PackageNotFoundError:  # pragma: nocover
    __version__ = "UNKNOWN"

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "operations": ["predict", "recommend", "score"],
        "pipeline": ["Pipeline", "RecPipelineBuilder", "topn_pipeline"],
    },
)
