# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing.context import SpawnContext, SpawnProcess
from multiprocessing.managers import SharedMemoryManager

import structlog
from typing_extensions import Any, Generic, Iterable, Iterator, override

from lenskit.logging.worker import WorkerContext, WorkerLogConfig

from . import worker
from .config import get_parallel_config
from .invoker import A, InvokeOp, M, ModelOpInvoker, R
from .serialize import shm_serialize

_log = structlog.stdlib.get_logger(__name__)


class ProcessPoolOpInvoker(ModelOpInvoker[A, R], Generic[M, A, R]):
    manager: SharedMemoryManager
    pool: ProcessPoolExecutor

    def __init__(self, model: M, func: InvokeOp[M, A, R], n_jobs: int):
        _log.debug("persisting function")
        ctx = LenskitMPContext()
        _log.info("setting up process pool w/ %d workers", n_jobs)
        kid_tc = get_parallel_config().child_threads

        self.manager = SharedMemoryManager()
        self.manager.start()

        try:
            cfg = worker.WorkerConfig(kid_tc)
            job = worker.WorkerData(func, model)
            job = shm_serialize(job, self.manager)
            self.pool = ProcessPoolExecutor(n_jobs, ctx, worker.initalize, (cfg, job))
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
        self.pool.shutdown()
        self.manager.shutdown()


class LenskitProcess(SpawnProcess):
    "LensKit worker process implementation."

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # save the log config to pass to the process
        self._log_config = WorkerLogConfig.current()

    @override
    def run(self):
        with WorkerContext(self._log_config):
            log = _log.bind(pid=self.pid, pname=self.name)
            log.info("multiprocessing worker started")
            super().run()


class LenskitMPContext(SpawnContext):
    "LensKit multiprocessing context."

    Process = LenskitProcess
