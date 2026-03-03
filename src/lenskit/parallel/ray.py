# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Support for parallelism with Ray.
"""

from __future__ import annotations

import base64
import os
import pickle
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator
from platform import python_version_tuple
from typing import Any, TypeVar, overload

import torch

from lenskit.logging import get_logger
from lenskit.logging.worker import WorkerContext, WorkerLogConfig

from .config import (
    effective_cpu_count,
    ensure_parallel_init,
    get_parallel_config,
)

try:
    import ray
    import ray.util
    from ray.remote_function import RemoteFunction

    _ray_imported = True
except ImportError:
    _ray_imported = False


T = TypeVar("T")

BATCH_SIZE = 200
_serializers_registered = False
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

    init_serializers()


def init_cluster(
    *,
    num_cpus: int | None = None,
    resources: dict[str, float] | None = None,
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

    if num_cpus is None:
        num_cpus = effective_cpu_count()

    env = cfg.env_vars().copy()
    wc = WorkerLogConfig.current()
    env["LK_LOG_CONFIG"] = base64.encodebytes(pickle.dumps(wc)).decode()

    setup = _worker_setup if global_logging else None
    runtime = ray.runtime_env.RuntimeEnv(env_vars=env, worker_process_setup_hook=setup)

    _log.info("starting Ray cluster")
    ray.init(num_cpus=num_cpus, resources=resources, runtime_env=runtime, **kwargs)


def init_serializers():
    global _serializers_registered
    if _serializers_registered:
        return

    _log.debug("registering Pytorch serializer")
    ray.util.register_serializer(
        torch.Tensor, serializer=_SharedTensor, deserializer=torch.as_tensor
    )

    _serializers_registered = True


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
            self.limit = get_parallel_config().resolved_num_batch_jobs
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

    def throttle[Elt](
        self, tasks: Generator[ray.ObjectRef[Elt]], *, ordered: bool = True
    ) -> Generator[Elt, None, int]:
        """
        Throttle a generator of Ray tasks, only requesting new tasks from the
        generator as the limit has space.

        Args:
            tasks:
                A generator of tasks (usually the result of calling a Ray remote
                function).
        Returns:
            A generator of the results (already awaited with :func:`ray.get`).
        """
        if ordered:
            return self._throttle_ordered(tasks)
        else:
            return self._throttle_unordered(tasks)

    @overload
    def imap(
        self,
        function: RemoteFunction,
        items: Iterable[Any],
        *,
        ordered: bool = True,
    ) -> Generator[Any, None, int]: ...
    @overload
    def imap[A, R](
        self,
        function: Callable[[A], R],
        items: Iterable[A],
        *,
        ordered: bool = True,
    ) -> Generator[R, None, int]: ...
    def imap[A, R](
        self,
        function: RemoteFunction | Callable[[A], R],
        items: Iterable[A],
        *,
        ordered: bool = True,
    ) -> Generator[R, None, int]:
        # make a callable from a Ray remote function
        if hasattr(function, "remote"):
            function = function.remote  # type: ignore

        tasks = (function(e) for e in items)
        return self.throttle(tasks)  # type: ignore

    def _throttle_ordered[Elt](
        self, tasks: Generator[ray.ObjectRef[Elt]]
    ) -> Generator[Elt, None, int]:
        n = 0
        queued = deque()
        ready = set()

        while True:
            for res in self.results_until_limit():
                ready.add(res)
                n += 1

            yield from _drain_queue(queued, ready)

            # now we invoke the next task
            try:
                task = next(tasks)
            except StopIteration:
                break
            self.add_task(task)
            queued.append(task)

        for res in self.drain_results():
            ready.add(res)
            n += 1

            yield from _drain_queue(queued, ready)

        assert len(queued) == 0
        assert len(ready) == 0

        return n

    def _throttle_unordered[Elt](
        self, tasks: Generator[ray.ObjectRef[Elt]]
    ) -> Generator[Elt, None, int]:
        n = 0

        while True:
            for res in self.results_until_limit():
                yield ray.get(res)
                n += 1

            # now we invoke the next task
            try:
                task = next(tasks)
            except StopIteration:
                break
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


class _SharedTensor:
    tensor: torch.Tensor

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __reduce__(self):
        from torch.multiprocessing.reductions import reduce_tensor

        return reduce_tensor(self.tensor)
