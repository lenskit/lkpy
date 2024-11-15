# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Callable, Generic, Iterable, Iterator, Optional, TypeAlias, TypeVar

from progress_api import Progress, make_progress

from .config import ensure_parallel_init, get_parallel_config

M = TypeVar("M")
A = TypeVar("A")
R = TypeVar("R")
InvokeOp: TypeAlias = Callable[[M, A], R]


def invoke_progress(
    logger: str | Logger | None = None,
    label: str | None = None,
    total: int | None = None,
    unit: str | None = None,
) -> Progress:
    """
    Create a progress bar for parallel tasks.  It is populated with the
    correct state of tasks for :func:`invoker`.

    See :func:`make_progress` for details on parameter meanings.
    """
    return make_progress(
        logger, label, total, outcomes="finished", states=["in-progress", "dispatched"], unit=unit
    )


def invoker(
    model: M,
    func: InvokeOp[M, A, R],
    n_jobs: Optional[int] = None,
    progress: Progress | None = None,
) -> ModelOpInvoker[A, R]:
    """
    Get an appropriate invoker for performing operations on ``model``.

    Args:
        model: The model object on which to perform operations.
        func: The function to call.  The function must be pickleable.
        n_jobs:
            The number of processes to use for parallel operations.  If ``None``, will
            call :func:`proc_count` with a maximum default process count of 4.
        progress:
            A progress bar to use to report status. It should have the following states:

            * dispatched
            * in-progress
            * finished

            One can be created with :func:`invoke_progress`

    Returns:
        ModelOpInvoker:
            An invoker to perform operations on the model.
    """
    ensure_parallel_init()
    if n_jobs is None:
        n_jobs = get_parallel_config().processes

    if n_jobs == 1:
        from .sequential import InProcessOpInvoker

        return InProcessOpInvoker(model, func, progress)
    else:
        from .pool import ProcessPoolOpInvoker

        return ProcessPoolOpInvoker(model, func, n_jobs, progress)


class ModelOpInvoker(ABC, Generic[A, R]):
    """
    Interface for invoking operations on a model, possibly in parallel.  The operation
    invoker is configured with a model and a function to apply, and applies that function
    to the arguments supplied in `map`.  Child process invokers also route logging messages
    to the parent process, so logging works even with multiprocessing.

    An invoker is a context manager that calls :meth:`shutdown` when exited.
    """

    @abstractmethod
    def map(self, tasks: Iterable[A]) -> Iterator[R]:
        """
        Apply the configured function to the model and iterables.  This is like
        :func:`map`, except it supplies the invoker's model as the first object
        to ``func``.

        Args:
            iterables: Iterables of arguments to provide to the function.

        Returns:
            iterable: An iterable of the results.
        """
        pass

    def shutdown(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args: Any):
        self.shutdown()
