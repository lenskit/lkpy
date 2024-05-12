# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
import functools as ft
import logging
from typing import Generic, Iterable, Iterator

from progress_api import Progress

from .invoker import A, InvokeOp, M, ModelOpInvoker, R

_log = logging.getLogger(__name__)


class InProcessOpInvoker(ModelOpInvoker[A, R], Generic[M, A, R]):
    model: M
    function: InvokeOp[M, A, R]

    def __init__(self, model: M, func: InvokeOp[M, A, R], progress: Progress | None = None):
        _log.info("setting up in-process worker")
        self.model = model
        self.function = func

    def map(self, tasks: Iterable[A]) -> Iterator[R]:
        proc = ft.partial(self.function, self.model)
        return map(proc, tasks)

    def shutdown(self):
        del self.model
