# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
from typing import Any

import manylog
import seedbank
import torch
from numpy.random import SeedSequence
from threadpoolctl import threadpool_limits
from typing_extensions import Generic, NamedTuple

from .invoker import A, InvokeOp, M, R
from .serialize import shm_deserialize

_log = logging.getLogger(__name__)


__work_context: WorkerContext


class WorkerConfig(NamedTuple):
    threads: int
    seed: SeedSequence
    log_addr: str


class WorkerContext(NamedTuple, Generic[M, A, R]):
    func: InvokeOp[M, A, R]
    model: M


def initalize(cfg: WorkerConfig, ctx: bytes) -> None:
    global __work_context

    manylog.init_worker_logging(cfg.log_addr)
    seed = seedbank.derive_seed(mp.current_process().name, base=cfg.seed)
    seedbank.initialize(seed)

    # disable BLAS threading
    threadpool_limits(limits=cfg.threads, user_api="blas")
    torch.set_num_interop_threads(cfg.threads)

    __work_context = shm_deserialize(ctx)

    _log.debug("worker %d ready (process %s)", os.getpid(), mp.current_process())


def worker(arg: Any) -> Any:
    res = __work_context.func(__work_context.model, arg)
    return pickle.dumps(res, pickle.HIGHEST_PROTOCOL)
