# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
from typing import Mapping

from lenskit.data import ID, GenericKey, ItemList, ItemListCollection
from lenskit.pipeline import Pipeline

from ._runner import BatchPipelineRunner

_logger = logging.getLogger(__name__)


def predict(
    pipeline: Pipeline,
    test: ItemListCollection[GenericKey] | Mapping[ID, ItemList],
    *,
    n_jobs: int | None = None,
    **kwargs,
) -> ItemListCollection[GenericKey]:
    """
    Convenience function to batch-generate rating predictions (or other per-item
    scores) from a pipeline.
    """

    runner = BatchPipelineRunner(n_jobs=n_jobs)
    runner.predict()
    outs = runner.run(pipeline, test)
    return outs.output("predictions")  # type: ignore
