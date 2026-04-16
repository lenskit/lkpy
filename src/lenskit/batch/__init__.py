# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Batch-run recommendation pipelines for evaluation.
"""

from __future__ import annotations

from typing import overload

from lenskit.data import GenericKey, ItemListCollection
from lenskit.pipeline import Pipeline, PipelineProfiler

from ._queries import BatchInput, BatchRecRequest, TestRequestAdapter
from ._results import BatchResults
from ._runner import BatchPipelineRunner, InvocationSpec

__all__ = [
    "BatchPipelineRunner",
    "BatchResults",
    "BatchRecRequest",
    "BatchInput",
    "TestRequestAdapter",
    "InvocationSpec",
    "predict",
    "recommend",
]


@overload
def predict[K: GenericKey](
    pipeline: Pipeline,
    test: ItemListCollection[K],
    *,
    n_jobs: int | None = None,
    use_ray: bool | None = None,
) -> ItemListCollection[K]: ...
@overload
def predict(
    pipeline: Pipeline,
    test: BatchInput,
    *,
    n_jobs: int | None = None,
    use_ray: bool | None = None,
) -> ItemListCollection[GenericKey]: ...
def predict(
    pipeline: Pipeline,
    test: BatchInput,
    *,
    n_jobs: int | None = None,
    use_ray: bool | None = None,
) -> ItemListCollection[GenericKey]:
    """
    Convenience function to batch-generate rating predictions (or other per-item
    scores) from a pipeline.  This is a batch version of
    :func:`lenskit.predict`, and is a convenience wrapper around using a
    :meth:`BatchPipelineRunner` to generate predictions.

    .. note::

        If ``test`` is just a sequence of IDs, this method will still work, but
        it will score _all candidate items_ for each of the IDs.

    Stability:
        Caller
    """

    runner = BatchPipelineRunner(n_jobs=n_jobs, use_ray=use_ray)
    runner.predict()
    outs = runner.run(pipeline, test)
    return outs.output("predictions")  # type: ignore


@overload
def score[K: GenericKey](
    pipeline: Pipeline,
    test: ItemListCollection[K],
    *,
    n_jobs: int | None = None,
    use_ray: bool | None = None,
) -> ItemListCollection[K]: ...
@overload
def score(
    pipeline: Pipeline,
    test: BatchInput,
    *,
    n_jobs: int | None = None,
    use_ray: bool | None = None,
) -> ItemListCollection[GenericKey]: ...
def score(
    pipeline: Pipeline,
    test: BatchInput,
    *,
    n_jobs: int | None = None,
    use_ray: bool | None = None,
) -> ItemListCollection[GenericKey]:
    """
    Convenience function to batch-generate personalized scores from a pipeline.
    This is a batch version of :func:`lenskit.predict`, and is a convenience
    wrapper around using a :meth:`BatchPipelineRunner` to generate item scores.

    .. note::

        If ``test`` is just a sequence of IDs, this method will still work, but
        it will score _all candidate items_ for each of the IDs.

    Stability:
        Caller
    """

    runner = BatchPipelineRunner(n_jobs=n_jobs, use_ray=use_ray)
    runner.score()
    outs = runner.run(pipeline, test)
    return outs.output("scores")  # type: ignore


@overload
def recommend[K: GenericKey](
    pipeline: Pipeline,
    queries: ItemListCollection[K],
    n: int | None = None,
    *,
    n_jobs: int | None = None,
    use_ray: bool | None = None,
    profiler: PipelineProfiler | None = None,
    users=None,
) -> ItemListCollection[K]: ...
@overload
def recommend(
    pipeline: Pipeline,
    queries: BatchInput,
    n: int | None = None,
    *,
    n_jobs: int | None = None,
    use_ray: bool | None = None,
    profiler: PipelineProfiler | None = None,
    users=None,
) -> ItemListCollection[GenericKey]: ...
def recommend(
    pipeline: Pipeline,
    queries: BatchInput,
    n: int | None = None,
    *,
    n_jobs: int | None = None,
    use_ray: bool | None = None,
    profiler: PipelineProfiler | None = None,
    users=None,
) -> ItemListCollection[GenericKey]:
    """
    Convenience function to batch-generate recommendations from a pipeline. This
    is a batch version of :func:`lenskit.recommend`, and is a convenience
    wrapper around using a :meth:`BatchPipelineRunner` to generate
    recommendations.

    .. seealso::

        :meth:`BatchPipelineRunner.run` for details on the arguments, and
        :ref:`batch-queries` for details on the valid inputs for ``queries``.

    Args:
        queries:
            The request queries.

    Stability:
        Caller
    """
    if users is not None:
        if queries is not None:
            raise RuntimeError("cannot pass both queries= and users=")
        queries = users

    runner = BatchPipelineRunner(n_jobs=n_jobs, use_ray=use_ray, profiler=profiler)
    runner.recommend(n=n)
    outs = runner.run(pipeline, queries)
    return outs.output("recommendations")  # type: ignore
