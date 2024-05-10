# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Generic, Iterable, Iterator

import manylog
import seedbank

from . import worker
from .config import proc_count
from .invoker import A, InvokeOp, M, ModelOpInvoker, R
from .serialize import init_reductions, shm_serialize

_log = logging.getLogger(__name__)
_log_listener: manylog.LogListener | None = None


class ProcessPoolOpInvoker(ModelOpInvoker[A, R], Generic[M, A, R]):
    pool: ProcessPoolExecutor

    def __init__(self, model: M, func: InvokeOp[M, A, R], n_jobs: int):
        _log.debug("persisting function")
        ctx = mp.get_context("spawn")
        _log.info("setting up process pool w/ %d workers", n_jobs)
        kid_tc = proc_count(level=1)
        seed = seedbank.root_seed()
        log_addr = ensure_log_listener()

        cfg = worker.WorkerConfig(kid_tc, seed, log_addr)
        job = worker.WorkerContext(func, model)
        job = shm_serialize(job)
        self.pool = ProcessPoolExecutor(n_jobs, ctx, worker.initalize, (cfg, job))

    def map(self, tasks: Iterable[A]) -> Iterator[R]:
        return self.pool.map(worker.worker, tasks)

    def shutdown(self):
        self.pool.shutdown()


def ensure_log_listener() -> str:
    global _log_listener
    if _log_listener is None:
        _log_listener = manylog.LogListener()
        _log_listener.start()

    addr = _log_listener.address
    assert addr is not None
    return addr


init_reductions()
