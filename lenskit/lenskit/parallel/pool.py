# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: basic
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing.context import SpawnContext, SpawnProcess
from multiprocessing.managers import SharedMemoryManager

from typing_extensions import Any, Generic, Iterable, Iterator, override

from lenskit.logging import Task, get_logger
from lenskit.logging.worker import WorkerContext, WorkerLogConfig

from . import worker
from .config import ParallelConfig, ensure_parallel_init, get_parallel_config, initialize
from .invoker import A, InvokeOp, M, ModelOpInvoker, R
from .serialize import shm_serialize

_log = get_logger(__name__)


def multiprocess_executor(
    n_jobs: int | None = None, sp_config: ParallelConfig | None = None
) -> ProcessPoolExecutor:
    """
    Construct a :class:`ProcessPoolExecutor` configured for LensKit work.
    """
    ensure_parallel_init()
    cfg = get_parallel_config()
    if sp_config is None:
        sp_config = ParallelConfig(1, 1, cfg.child_threads, 1)
    if n_jobs is None:
        n_jobs = cfg.processes

    ctx = LensKitMPContext(sp_config)
    return ProcessPoolExecutor(n_jobs, ctx)


class ProcessPoolOpInvoker(ModelOpInvoker[A, R], Generic[M, A, R]):
    manager: SharedMemoryManager
    pool: ProcessPoolExecutor

    def __init__(
        self,
        model: M,
        func: InvokeOp[M, A, R],
        n_jobs: int,
        worker_parallel: ParallelConfig | None = None,
    ):
        log = _log.bind(n_jobs=n_jobs)
        log.debug("persisting function")
        if worker_parallel is None:
            worker_parallel = ParallelConfig(1, 1, get_parallel_config().child_threads, 1)
        ctx = LensKitMPContext(worker_parallel)

        log.debug("initializing shared memory")
        self.manager = SharedMemoryManager()
        self.manager.start()

        try:
            job = worker.WorkerData(func, model)
            job = shm_serialize(job, self.manager)
            log.info("setting up process pool")
            self.pool = ProcessPoolExecutor(n_jobs, ctx, worker.initalize, (job,))
        except Exception as e:
            self.manager.shutdown()
            raise e

    def map(self, tasks: Iterable[A]) -> Iterator[R]:
        return self.pool.map(worker.worker, self._task_iter(tasks))

    def _task_iter(self, tasks: Iterable[A]):
        """
        Yield the tasks, recording each as dispatched before it is yielded.
        """
        for task in tasks:
            yield task

    def shutdown(self):
        _log.debug("shutting down process pool")
        self.pool.shutdown()
        self.manager.shutdown()
        _log.debug("process pool shut down")


class LensKitProcess(SpawnProcess):
    "LensKit worker process implementation."

    def __init__(
        self, logging: WorkerLogConfig, parallel: ParallelConfig, *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        # save the log config to pass to the process
        self._log_config = logging
        self._parallel_config = parallel

    @override
    def run(self):
        with WorkerContext(self._log_config) as ctx:
            initialize(self._parallel_config)
            log = _log.bind(pid=self.pid, pname=self.name)
            log.info("multiprocessing worker started")
            task = None
            try:
                with Task("worker process", subprocess=True) as task:
                    ctx.send_task(task)
                    super().run()
            finally:
                if task is not None:
                    ctx.send_task(task)


class LensKitMPContext(SpawnContext):
    "LensKit multiprocessing context."

    _log_config: WorkerLogConfig
    _parallel_config: ParallelConfig

    def __init__(self, parallel: ParallelConfig):
        self._log_config = WorkerLogConfig.current()
        self._parallel_config = parallel

    def Process(self, *args: Any, **kwargs: Any) -> SpawnProcess:
        return LensKitProcess(self._log_config, self._parallel_config, *args, **kwargs)
