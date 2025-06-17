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
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from functools import partial
from platform import python_version_tuple
from typing import Any, Generic, TypeVar

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

try:
    import ray
    from ray.remote_function import RemoteFunction

    _ray_imported = True
except ImportError:
    _ray_imported = False


T = TypeVar("T")

BATCH_SIZE = 200
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
    global_logging: bool = True,
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
    import ray.runtime_env

    if resources is None:
        resources = {}
    else:
        resources = resources.copy()

    cfg = get_parallel_config()
    if proc_slots is None:
        proc_slots = cfg.processes

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

    _log.info("starting Ray cluster", logging=global_logging)
    ray.init(num_cpus=num_cpus, resources=resources, runtime_env=runtime, **kwargs)


def inference_worker_cpus() -> int:
    return _worker_parallel.backend_threads


def training_worker_cpus() -> int:
    return _worker_parallel.total_threads


def ray_supported() -> bool:
    """
    Check if this Ray setup is supported by LensKit.
    """
    major, minor, patch = python_version_tuple()
    if int(major) < 3 or int(minor) < 12:
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
        ctx = ray.get_runtime_context()
        return ctx.worker.mode == ray.WORKER_MODE
    else:
        return False


class RayOpInvoker(ModelOpInvoker[A, R], Generic[M, A, R]):
    function: InvokeOp[M, A, R]
    model_ref: Any
    action: RemoteFunction
    limit: int | None

    def __init__(self, model: M, func: InvokeOp[M, A, R], *, limit: int | None = None):
        _log.debug("persisting to Ray cluster")
        import ray
        import torch

        if isinstance(model, ray.ObjectRef):
            self.model_ref = model
        else:
            self.model_ref = ray.put(model)
        self.function = func

        self.limit = limit

        worker = ray.remote(_ray_invoke_worker)
        # request a little GPU
        if torch.cuda.is_available():
            n_gpus = 0.1
        else:
            n_gpus = None
        self.action = worker.options(num_cpus=inference_worker_cpus(), num_gpus=n_gpus)  # type: ignore

    def map(self, tasks: Iterable[A]) -> Iterator[R]:
        limit = TaskLimiter(self.limit)
        function = partial(self.action.remote, self.function, self.model_ref)
        for bres in limit.imap(function, itertools.batched(tasks, BATCH_SIZE)):
            yield from bres  # type: ignore

    def shutdown(self):
        del self.model_ref


class TaskLimiter:
    """
    Limit task concurrency using :func:`ray.wait`.

    This class provides two key operations:

    - Add a task to the limiter with :meth:`add_task`.
    - Wait for tasks until the number of pending tasks is less than the limit.
    - Wait for all tasks with `drain`.

    Args:
        limit:
            The maximum number of pending tasks.  Defaults to the LensKit
            process count (see :func:`lenskit.parallel.initialize`).
    """

    limit: int
    "The maximum number of pending tasks."
    finished: int
    "The number of tasks completed."

    _tasks: list[Any]

    def __init__(self, limit: int | None = None):
        if limit is None or limit <= 0:
            self.limit = get_parallel_config().processes
        else:
            self.limit = limit
        self.finished = 0
        self._tasks = []

    @property
    def pending(self) -> int:
        """
        The number of pending tasks.
        """
        return len(self._tasks)

    def imap(
        self,
        function: RemoteFunction | Callable[[A], R],
        items: Iterable[A],
        *,
        ordered: bool = True,
    ) -> Generator[R, None, int]:
        if ordered:
            return self._imap_ordered(function, items)
        else:
            return self._imap_unordered(function, items)

    def _imap_ordered(
        self, function: RemoteFunction | Callable[[A], R], items: Iterable[A]
    ) -> Generator[Any, None, int]:
        n = 0
        queued = deque()
        ready = set()

        # make a callable from a Ray remote function
        if hasattr(function, "remote"):
            function = function.remote  # type: ignore

        for arg in items:
            for res in self.results_until_limit():
                ready.add(res)
                n += 1

            yield from _drain_queue(queued, ready)

            task: Any = function(arg)
            self.add_task(task)
            queued.append(task)

        for res in self.drain_results():
            ready.add(res)
            n += 1

            yield from _drain_queue(queued, ready)

        assert len(queued) == 0
        assert len(ready) == 0

        return n

    def _imap_unordered(
        self, function: RemoteFunction | Callable[[A], R], items: Iterable[A]
    ) -> Generator[Any, None, int]:
        n = 0

        # make a callable from a Ray remote function
        if hasattr(function, "remote"):
            function = function.remote  # type: ignore

        for arg in items:
            for res in self.results_until_limit():
                yield ray.get(res)
                n += 1
            task: Any = function(arg)
            self.add_task(task)

        for res in self.drain_results():
            yield ray.get(res)
            n += 1

        return n

    def add_task(self, task: ray.ObjectRef | ray.ObjectID):
        self._tasks.append(task)

    def results_until_limit(self) -> Generator[ray.ObjectRef, None, None]:
        """
        Iterate over available results until the number of pending results tasks
        is less than the limit, blocking as needed.

        This is a generator, returning the task result references.  The iterator
        will stop when the pending tasks list is under the limit.  No guarantee
        is made on the order of returned results.
        """
        return self._drain_until(self.limit - 1)

    def wait_for_limit(self):
        """
        Wait until the pending tasks are back under the limit.

        This method calls :meth:`ray.get` on the result of each pending task to
        resolve errors, but discards the return value.
        """
        for r in self.results_until_limit():
            ray.get(r)

    def drain_results(self) -> Generator[ray.ObjectRef, None, None]:
        """
        Iterate over all remaining tasks until the pending task list is empty,
        blocking as needed.

        This is a generator, returning the task result references.  No guarantee
        is made on the order of returned results.
        """
        return self._drain_until(0)

    def drain(self):
        """
        Wait until all pending tasks are finished.

        This method calls :meth:`ray.get` on the result of each pending task to
        resolve errors, but discards the return value.
        """
        for r in self.drain_results():
            ray.get(r)

    def _drain_until(self, limit: int) -> Generator[ray.ObjectRef, None, None]:
        while self._tasks and len(self._tasks) > limit:
            done, remaining = ray.wait(self._tasks)
            self._tasks = remaining
            for dt in done:
                self.finished += 1
                yield dt


def _drain_queue(queued: deque[ray.ObjectRef], ready: set[ray.ObjectRef]) -> Iterator[Any]:
    while queued and queued[0] in ready:
        ref = queued.popleft()
        ready.remove(ref)
        yield ray.get(ref)


def _ray_invoke_worker(func: Callable[[M, A], R], model: M, args: Sequence[A]) -> list[R]:
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

        _log.debug("worker context initialized")

    ensure_parallel_init()
    return context
