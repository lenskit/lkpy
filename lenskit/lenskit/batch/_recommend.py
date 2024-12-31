# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
from typing import Sequence

from lenskit.data import ID, ItemListCollection, UserIDKey
from lenskit.pipeline import Pipeline

from ._runner import BatchPipelineRunner

_logger = logging.getLogger(__name__)


def recommend(
    pipeline: Pipeline,
    users: Sequence[ID | UserIDKey],
    n: int | None = None,
    candidates=None,
    *,
    n_jobs: int | None = None,
    **kwargs,
) -> ItemListCollection[UserIDKey]:
    """
    Convenience function to batch-generate recommendations from a pipeline.

    Stability:
        Caller
    """

    runner = BatchPipelineRunner(n_jobs=n_jobs)
    runner.recommend(n=n)
    outs = runner.run(pipeline, users)
    return outs.output("recommendations")
