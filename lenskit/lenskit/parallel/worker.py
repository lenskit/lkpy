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

from typing_extensions import Generic

from lenskit.logging import get_logger

from .invoker import A, InvokeOp, M, R
from .serialize import SHMData, shm_deserialize

_log = get_logger(__name__)

__work_context: WorkerData


@dataclass(frozen=True)
class WorkerData(Generic[M, A, R]):
    func: InvokeOp[M, A, R]
    model: M


def initalize(ctx: SHMData) -> None:
    global __work_context, __progress
    proc = mp.current_process()
    log = _log.bind(pid=proc.pid, pname=proc.name)

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
