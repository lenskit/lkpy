# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Recommender systems toolkit.
"""

import lazy_loader as lazy

from ._version import lenskit_version

__version__ = lenskit_version()


# lazy-load LensKit internal imports (per SPEC-1)
# IMPORTANT: this must be kept in sync with __init__.pyi
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        # modules
        "batch",
        "config",
        "data",
        "logging",
        "metrics",
        "pipeline",
        "operations",
        "splitting",
        "stats",
        "torch",
        "training",
        # component modules
        "als",
        "basic",
        "flexmf",
        "funksvd",
        "hpf",
        "knn",
        "sklearn",
    ],
    submod_attrs={
        "config": ["configure", "lenskit_config"],
        "data": ["Dataset", "DatasetBuilder"],
        "operations": ["predict", "recommend", "score"],
        "pipeline": ["Pipeline", "RecPipelineBuilder", "Component", "topn_pipeline"],
        "splitting": ["TTSplit"],
    },
)
