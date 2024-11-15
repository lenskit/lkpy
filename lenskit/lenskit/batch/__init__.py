# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Batch-run recommendation pipelines for evaluation.
"""

from __future__ import annotations

from ._predict import predict
from ._recommend import recommend
from ._results import BatchResults
from ._runner import BatchPipelineRunner, InvocationSpec

__all__ = ["BatchPipelineRunner", "BatchResults", "InvocationSpec", "predict", "recommend"]
