# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Recommender systems toolkit.
"""

from importlib.metadata import PackageNotFoundError, version

from .operations import predict, recommend, score
from .pipeline import Pipeline, RecPipelineBuilder, topn_pipeline

__all__ = ["predict", "recommend", "score", "Pipeline", "RecPipelineBuilder", "topn_pipeline"]

try:
    __version__ = version("lenskit")
except PackageNotFoundError:  # pragma: nocover
    __version__ = "UNKNOWN"
