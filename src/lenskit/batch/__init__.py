# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Batch-run recommendation pipelines for evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Mapping

import pandas as pd

from lenskit.data import ID, GenericKey, ItemList, ItemListCollection, UserIDKey
from lenskit.pipeline import Pipeline

from ._results import BatchResults
from ._runner import BatchPipelineRunner, InvocationSpec

__all__ = ["BatchPipelineRunner", "BatchResults", "InvocationSpec", "predict", "recommend"]


def predict(
    pipeline: Pipeline,
    test: ItemListCollection[GenericKey] | Mapping[ID, ItemList] | pd.DataFrame,
    *,
    n_jobs: int | Literal["ray"] | None = None,
) -> ItemListCollection[GenericKey]:
    """
    Convenience function to batch-generate rating predictions (or other per-item
    scores) from a pipeline.  This is a batch version of :func:`lenskit.predict`.

    .. note::

        If ``test`` is just a sequence of IDs, this method will still work, but
        it will score _all candidate items_ for each of the IDs.

    Stability:
        Caller
    """

    runner = BatchPipelineRunner(n_jobs=n_jobs)
    runner.predict()
    outs = runner.run(pipeline, test)
    return outs.output("predictions")  # type: ignore


def score(
    pipeline: Pipeline,
    test: ItemListCollection[GenericKey] | Mapping[ID, ItemList] | pd.DataFrame,
    *,
    n_jobs: int | Literal["ray"] | None = None,
) -> ItemListCollection[GenericKey]:
    """
    Convenience function to batch-generate personalized scores from a pipeline.
    This is a batch version of :func:`lenskit.predict`.

    .. note::

        If ``test`` is just a sequence of IDs, this method will still work, but
        it will score _all candidate items_ for each of the IDs.

    Stability:
        Caller
    """

    runner = BatchPipelineRunner(n_jobs=n_jobs)
    runner.score()
    outs = runner.run(pipeline, test)
    return outs.output("scores")  # type: ignore


def recommend(
    pipeline: Pipeline,
    users: ItemListCollection[GenericKey]
    | Mapping[ID, ItemList]
    | Iterable[ID | GenericKey]
    | pd.DataFrame,
    n: int | None = None,
    *,
    n_jobs: int | Literal["ray"] | None = None,
) -> ItemListCollection[UserIDKey]:
    """
    Convenience function to batch-generate recommendations from a pipeline. This
    is a batch version of :func:`lenskit.recommend`.

    .. todo::

        Support more inputs than just user IDs.

    Stability:
        Caller
    """

    runner = BatchPipelineRunner(n_jobs=n_jobs)
    runner.recommend(n=n)
    outs = runner.run(pipeline, users)
    return outs.output("recommendations")  # type: ignore
