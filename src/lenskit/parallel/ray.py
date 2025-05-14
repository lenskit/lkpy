# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Support for parallelism with Ray.
"""

from __future__ import annotations

import base64
import itertools
import os
import pickle
from collections.abc import Callable, Iterable, Iterator
from platform import python_version
from typing import Any, Generic

from lenskit.logging import Task, get_logger
from lenskit.logging.worker import WorkerContext, WorkerLogConfig

from .config import (
    ParallelConfig,
    effective_cpu_count,
    ensure_parallel_init,
    get_parallel_config,
    subprocess_config,
)
from .invoker import A, InvokeOp, M, ModelOpInvoker, R

LK_PROCESS_SLOT = "lk_process"
_worker_parallel: ParallelConfig
_log = get_logger(__name__)


def ensure_cluster():
    import ray

    if not ray.is_initialized():
        _log.debug("Ray is not initialized, initializing")
        try:
            init_cluster()
        except ValueError as e:
            _log.debug("Ray initialization failed", exception=e)
            if "existing cluster" in str(e):
                _log.info("Ray cluster already started, reusing")
                _log.warn("existing Ray cluster may not properly throttle LensKit jobs")
                ray.init()
            else:
                raise e


def init_cluster(
    *,
    num_cpus: int | None = None,
    proc_slots: int | None = None,
    resources: dict[str, float] | None = None,
    worker_parallel: ParallelConfig | None = None,
    limit_slots: bool = True,
    global_logging: bool = False,
    **kwargs,
):
    """
    Initialize or connect to a Ray cluster, with the LensKit options.

    The resulting cluster can be used by an invoker, or it can be used directly.
    The Ray invoker uses batching, though, so it only works well with many small
    tasks.

    Args:
        num_cpus:
            The total number of CPUs to allow. Defaults to
            :fun:`effective_cpu_count`.
        proc_slots:
            The number of “process slots” for LensKit parallel operations.
            Defaults to the LensKit process count.  These slots are recorded as
            the ``lk_process`` resource on the Ray cluster.
        resources:
            Additional custom resources to register in the Ray cluster.
        worker_parallel:
            Parallel processing configuration for worker processes.  If
            ``None``, uses the default.
        limit_slots:
            ``False`` to disable the LensKit slot interface.
        global_logging:
            ``True`` to wire up logging in the workers at startup, instead of only
            connecting logs when a task is run.
        kwargs:
            Other options to pass to :func:`ray.init`.

    Stability:
        Experimental
    """
    global _worker_parallel
    import ray

    if resources is None:
        resources = {}
    else:
        resources = resources.copy()

    cfg = get_parallel_config()
    if proc_slots is None:
        proc_slots = cfg.processes

    if limit_slots:
        resources[LK_PROCESS_SLOT] = proc_slots

    if num_cpus is None:
        num_cpus = effective_cpu_count()

    if worker_parallel is None:
        worker_parallel = subprocess_config()
    _worker_parallel = worker_parallel

    env = worker_parallel.env_vars().copy()
    wc = WorkerLogConfig.current()
    env["LK_LOG_CONFIG"] = base64.encodebytes(pickle.dumps(wc)).decode()

    setup = _worker_setup if global_logging else None
    runtime = ray.runtime_env.RuntimeEnv(env_vars=env, worker_process_setup_hook=setup)

    _log.info("starting Ray cluster")
    ray.init(num_cpus=num_cpus, resources=resources, runtime_env=runtime, **kwargs)


def inference_worker_cpus() -> int:
    return _worker_parallel.backend_threads


def training_worker_cpus() -> int:
    return _worker_parallel.total_threads


def ray_supported() -> bool:
    """
    Check if this Ray setup is supported by LensKit.
    """
    if python_version() < "3.12":
        return False
    else:
        return ray_available()


def ray_available() -> bool:
    """
    Check if Ray is available.
    """
    try:
        import ray  # noqa: F401

        return True
    except ImportError:
        return False


def ray_active() -> bool:
    """
    Query whether Ray is active.
    """
    if ray_available():
        import ray

        return ray.is_initialized()
    else:
        return False


def is_ray_worker() -> bool:
    """
    Determine whether the current process is running on a Ray worker.
    """
    # logic adapted from https://discuss.ray.io/t/how-to-know-if-code-is-running-on-ray-worker/15642
    if ray_active():
        import ray

        ctx = ray.get_runtime_context()
        return ctx.worker.mode == ray.WORKER_MODE
    else:
        return False


class RayOpInvoker(ModelOpInvoker[A, R], Generic[M, A, R]):
    function: InvokeOp[M, A, R]
    model_ref: Any

    def __init__(
        self,
        model: M,
        func: InvokeOp[M, A, R],
    ):
        _log.debug("persisting to Ray cluster")
        import ray

        if isinstance(model, ray.ObjectRef):
            self.model_ref = model
        else:
            self.model_ref = ray.put(model)
        self.function = func
        slots = {}
        if LK_PROCESS_SLOT in ray.cluster_resources():
            slots = {LK_PROCESS_SLOT: 1}
        else:
            _log.warning(f"cluster has no resource {LK_PROCESS_SLOT}")

        worker = ray.remote(_ray_invoke_worker)
        self.action = worker.options(num_cpus=inference_worker_cpus(), resources=slots)

    def map(self, tasks: Iterable[A]) -> Iterator[R]:
        import ray

        batch_results = [
            self.action.remote(self.function, self.model_ref, batch)
            for batch in itertools.batched(tasks, 200)
        ]
        for br in batch_results:
            yield from ray.get(br)

    def shutdown(self):
        del self.model_ref


def _ray_invoke_worker(func: Callable[[M, A], R], model: M, args: list[A]) -> list[R]:
    with init_worker(autostart=False) as ctx:
        try:
            with Task("cluster worker", subprocess=True) as task:
                result = [func(model, arg) for arg in args]
        finally:
            ctx.send_task(task)

    return result


def _worker_setup():
    init_worker()


def init_worker(*, autostart: bool = True) -> WorkerContext:
    """
    Initialize a Ray worker process.  Sets up logging, and returns the context.

    Args:
        autostart:
            Set to ``False`` to disable calling :meth:`WorkerContext.start`, for
            when the caller will start and stop the context if it is new.
    """
    log_cfg = pickle.loads(base64.decodebytes(os.environb[b"LK_LOG_CONFIG"]))
    context = WorkerContext.active()
    if context is None:
        context = WorkerContext(log_cfg)
        if autostart:
            context.start()

    ensure_parallel_init()
    return context
