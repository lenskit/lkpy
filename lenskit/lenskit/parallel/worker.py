# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import multiprocessing as mp
import os
import warnings
from dataclasses import dataclass
from typing import Any

import structlog
from typing_extensions import Generic

from .config import initialize as init_parallel
from .invoker import A, InvokeOp, M, R
from .serialize import ModelData, shm_deserialize

_log = structlog.get_logger(__name__)


__work_context: WorkerData


@dataclass(frozen=True)
class WorkerConfig:
    threads: int


@dataclass(frozen=True)
class WorkerData(Generic[M, A, R]):
    func: InvokeOp[M, A, R]
    model: M


def initalize(cfg: WorkerConfig, ctx: ModelData) -> None:
    global __work_context, __progress
    proc = mp.current_process()
    log = _log.bind(pid=proc.pid, pname=proc.name)
    init_parallel(processes=1, threads=1, backend_threads=cfg.threads, child_threads=1)

    warnings.filterwarnings("ignore", "Sparse CSR tensor support is in beta state", UserWarning)

    try:
        __work_context = shm_deserialize(ctx)
    except Exception as e:
        log.error("deserialization failed: %s", e)
        raise e

    log.debug("worker %d ready (process %s)", os.getpid(), mp.current_process().name)


def worker(arg: Any) -> Any:
    res = __work_context.func(__work_context.model, arg)
    return res
