# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import warnings
from typing import Any
from uuid import UUID

import manylog
from progress_api import Progress
from typing_extensions import Generic, NamedTuple

from .config import initialize as init_parallel
from .invoker import A, InvokeOp, M, R
from .serialize import ModelData, shm_deserialize

_log = logging.getLogger(__name__)


__work_context: WorkerContext
__progress: Progress


class WorkerConfig(NamedTuple):
    threads: int


class WorkerContext(NamedTuple, Generic[M, A, R]):
    func: InvokeOp[M, A, R]
    model: M
    progress: UUID


def initalize(cfg: WorkerConfig, ctx: ModelData) -> None:
    global __work_context, __progress
    manylog.initialize()
    init_parallel(processes=1, threads=1, backend_threads=cfg.threads, child_threads=1)

    warnings.filterwarnings("ignore", "Sparse CSR tensor support is in beta state", UserWarning)

    try:
        __work_context = shm_deserialize(ctx)
    except Exception as e:
        _log.error("deserialization failed: %s", e)
        raise e

    __progress = manylog.connect_progress(__work_context.progress)

    _log.debug("worker %d ready (process %s)", os.getpid(), mp.current_process())


def worker(arg: Any) -> Any:
    __progress.update(1, "in-progress", "dispatched")
    res = __work_context.func(__work_context.model, arg)
    __progress.update(1, "finished", "in-progress")
    return res
