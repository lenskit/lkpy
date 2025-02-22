# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Recommender systems toolkit.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "UNSPECIFIED"


from .operations import predict, recommend, score
from .pipeline import Pipeline, RecPipelineBuilder, topn_pipeline

__all__ = ["predict", "recommend", "score", "Pipeline", "RecPipelineBuilder", "topn_pipeline"]
