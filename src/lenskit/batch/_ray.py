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
from lenskit.parallel.ray import TaskLimiter, ensure_cluster, inference_worker_cpus
from lenskit.pipeline import Pipeline, ProfileSink

from ._queries import BatchRequest
from ._results import BatchResultRow
from ._runner import InvocationSpec, run_pipeline

_log = get_logger(__name__)


def ray_results(
    pipeline: Pipeline,
    profiler: ProfileSink | None,
    invocations: list[InvocationSpec],
    queries: Iterable[BatchRequest],
    batch_size: int,
) -> Generator[BatchResultRow]:
    ensure_cluster()
    pc = get_parallel_config()
    limit = TaskLimiter(pc.batch_jobs)
    batches = batched(queries, batch_size)

    if profiler is not None:
        profiler = profiler.multiprocess()

    pipe_h = ray.put(pipeline)

    worker = ray.remote(_run_batch).options(num_cpus=inference_worker_cpus())
    tasks = (worker.remote(pipe_h, invocations, profiler, batch) for batch in batches)

    _log.info("running inference queries on Ray")
    for res in limit.throttle(tasks):
        yield from res


def _run_batch(
    pipeline: Pipeline,
    invocations: list[InvocationSpec],
    profiler: ProfileSink | None,
    requests: Sequence[BatchRequest],
) -> list[BatchResultRow]:
    ctx = (pipeline, invocations, profiler)
    return [run_pipeline((ctx), req) for req in requests]
