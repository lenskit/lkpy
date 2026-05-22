# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Support for batch inference on Ray.
"""

from collections.abc import Generator, Iterable, Sequence
from itertools import batched

import ray

from lenskit.logging import get_logger
from lenskit.parallel import get_parallel_config
from lenskit.parallel.ray import TaskLimiter, ensure_cluster
from lenskit.pipeline import Pipeline, ProfileSink

from ._queries import ResolvedBatchRequest
from ._results import BatchResultRow
from ._runner import InvocationSpec, run_pipeline

_log = get_logger(__name__)


def ray_results(
    pipeline: Pipeline,
    profiler: ProfileSink | None,
    invocations: list[InvocationSpec],
    queries: Iterable[ResolvedBatchRequest],
    n_jobs: int,
    batch_size: int,
) -> Generator[BatchResultRow]:
    ensure_cluster()
    pc = get_parallel_config()
    if n_jobs <= 0:
        n_jobs = pc.num_cpus

    limit = TaskLimiter(n_jobs)
    batches = batched(queries, batch_size)

    if profiler is not None:
        profiler = profiler.multiprocess()

    pipe_h = ray.put(pipeline)

    worker = ray.remote(_run_batch).options(num_cpus=pc.num_backend_threads or 1)
    tasks = (worker.remote(pipe_h, invocations, profiler, batch) for batch in batches)

    _log.info("running inference queries on Ray")
    for res in limit.throttle(tasks):
        yield from res


def _run_batch(
    pipeline: Pipeline,
    invocations: list[InvocationSpec],
    profiler: ProfileSink | None,
    requests: Sequence[ResolvedBatchRequest],
) -> list[BatchResultRow]:
    return [run_pipeline(pipeline, invocations, profiler, req) for req in requests]
