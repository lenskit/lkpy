# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Iterable, Iterator, Literal, TypeAlias, TypeVar

from .config import ParallelConfig, ensure_parallel_init, get_parallel_config

M = TypeVar("M")
A = TypeVar("A")
R = TypeVar("R")
InvokeOp: TypeAlias = Callable[[M, A], R]


def invoker(
    model: M,
    func: InvokeOp[M, A, R],
    n_jobs: int | None | Literal["ray"] = None,
    *,
    worker_parallel: ParallelConfig | None = None,
) -> ModelOpInvoker[A, R]:
    """
    Get an appropriate invoker for performing operations on ``model``.

    Args:
        model:
            The model object on which to perform operations.
        func:
            The function to call.  The function must be pickleable.
        n_jobs:
            The number of processes to use for parallel operations.  If
            ``None``, will call :func:`proc_count` with a maximum default
            process count of 4.  If ``ray`` (experimental), uses the Ray
            cluster.
        worker_parallel:
            A parallel configuration for subprocess workers.

    Returns:
        ModelOpInvoker:
            An invoker to perform operations on the model.

    Stability:
        caller
    """
    ensure_parallel_init()
    if n_jobs is None:
        n_jobs = get_parallel_config().processes

    if n_jobs == "ray":
        from .ray import RayOpInvoker, ensure_cluster, ray_supported

        if not ray_supported():
            raise RuntimeError("ray backend not available")

        ensure_cluster()
        return RayOpInvoker(model, func)

    if n_jobs == 1:
        from .sequential import InProcessOpInvoker

        return InProcessOpInvoker(model, func)
    else:
        from .pool import ProcessPoolOpInvoker

        return ProcessPoolOpInvoker(model, func, n_jobs, worker_parallel=worker_parallel)


def set_backend(name: str):
    global _backend
    _backend = name


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
        """
        Shut down this invoker.
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args: Any):
        self.shutdown()
