# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: basic
from __future__ import annotations

import logging
import os
import warnings
from typing import Optional

import torch
from threadpoolctl import threadpool_limits

from lenskit._accel import init_accel_pool

_config: Optional[ParallelConfig] = None
_log = logging.getLogger(__name__)


class ParallelConfig:
    """
    Parallel processing configuration.
    """

    _processes: list[int]
    _threads: list[int]
    _backend_threads: list[int]

    def __init__(
        self, processes: int | list[int], threads: int | list[int], backend_threads: int | list[int]
    ):
        self._processes = _ensure_numlist(processes)
        self._threads = _ensure_numlist(threads)
        self._backend_threads = _ensure_numlist(backend_threads)

    @property
    def processes(self) -> int:
        """
        Get the number of workers this process may spawn.
        """
        return self._processes[0]

    @property
    def threads(self) -> int:
        """
        Get the number of direct threads for this worker.
        """
        return self._threads[0]

    @property
    def backend_threads(self) -> int:
        """
        Get the number of backend threads for this worker.
        """
        return self._backend_threads[0]

    @property
    def total_threads(self) -> int:
        """
        Get the total number of threads for this worker (training threads times
        backend threads).
        """
        return self.threads * self.backend_threads

    def env_vars(self) -> dict[str, str]:
        """
        Get this parallel configuration as a set of environment variables.  The
        set also includes ``OMP_NUM_THREADS`` to configure OMP early.
        """
        evs = {
            "LK_NUM_PROCS": str(self.processes),
            "LK_NUM_THREADS": str(self.threads),
            "LK_NUM_BACKEND_THREADS": str(self.backend_threads),
        }
        return evs

    def subprocess(
        self,
        processes: int | list[int] | None = None,
        threads: int | list[int] | None = None,
        backend_threads: int | list[int] | None = None,
    ) -> ParallelConfig:
        """
        Get the parallel configuration for a subprocess.
        """
        ncpus = effective_cpu_count()

        if processes is None:
            if len(self._processes) > 1:
                processes = self._processes[1:]
                np = processes[0]
            else:
                np = processes = 1

        if threads is None:
            if len(self._threads) > 1:
                threads = self._threads[1:]
            else:
                allowed = min(ncpus // np, 4)
                threads = min(allowed, self.threads)

        if backend_threads is None:
            if len(self._backend_threads) > 1:
                backend_threads = self._backend_threads[1:]
            else:
                allowed = min(ncpus // np, 4)
                backend_threads = min(allowed, self.backend_threads)

        return ParallelConfig(processes=processes, threads=threads, backend_threads=backend_threads)


def initialize(
    config: ParallelConfig | None = None,
    *,
    processes: int | list[int] | None = None,
    threads: int | list[int] | None = None,
    backend_threads: int | list[int] | None = None,
    child_threads: int | None = None,
):
    """
    Set up and configure LensKit parallelism.  This only needs to be called if
    you want to control when and how parallelism is set up; components using
    parallelism will call :func:`ensure_init`, which will call this function
    with its default arguments if it has not been called.

    .. seealso:: :ref:`parallel-config`

    Args:
        processes:
            The number of processes to use for multiprocessing evaluations (see
            :envvar:`LK_NUM_PROCS`)
        threads:
            The number of threads to use for parallel model training and similar
            operations (see :envvar:`LK_NUM_THREADS`).
        backend_threads:
            The number of threads underlying computational engines should use
            (see :envvar:`LK_NUM_BACKEND_THREADS`).
        child_threads:
            The number of threads backends are allowed to use in the worker
            processes in multiprocessing operations (see
            :envvar:`LK_NUM_CHILD_THREADS`). Deprecated.
    """
    global _config
    if _config:
        _log.warning("parallelism already initialized")
        return

    # our parallel computation doesn't work with FD sharing
    torch.multiprocessing.set_sharing_strategy("file_system")

    if config is None:
        _config = _resolve_parallel_config(processes, threads, backend_threads, child_threads)
    else:
        _config = config
    _log.debug("configuring for parallelism: %s", _config)

    if 'OPENBLAS_NUM_THREADS' not in os.environ and 'MKL_NUM_THREADS' not in os.environ:
        threadpool_limits(_config.backend_threads, "blas")
    try:
        torch.set_num_interop_threads(_config.threads)
    except RuntimeError as e:
        _log.warning("failed to configure Pytorch interop threads: %s", e)
        warnings.warn("failed to set interop threads", RuntimeWarning)

    try:
        init_accel_pool(_config.threads)
    except RuntimeError as e:
        _log.warning("failed to initialize Rayon backend: %s", e)

    try:
        torch.set_num_threads(_config.backend_threads)
    except RuntimeError as e:
        _log.warning("failed to configure Pytorch intra-op threads: %s", e)
        warnings.warn("failed to set intra-op threads", RuntimeWarning)


def ensure_parallel_init():
    """
    Make sure LensKit parallelism is configured, and configure with defaults if
    it is not.

    Components using parallelism or intensive computations should call this
    function before they begin training.
    """
    if not _config:
        initialize()


def get_parallel_config() -> ParallelConfig:
    """
    Ensure that parallelism is configured and return the configuration.
    """
    ensure_parallel_init()
    assert _config is not None
    return _config


def subprocess_config(
    processes: int | list[int] | None = None,
    threads: int | list[int] | None = None,
    backend_threads: int | list[int] | None = None,
) -> ParallelConfig:
    """
    Get a parallel configuration for a subprocess.
    """
    cfg = get_parallel_config()
    return cfg.subprocess(processes, threads, backend_threads)


def effective_cpu_count() -> int:
    """
    Return the effective CPU count using the best available data.  Tries the following in order:

    1.  :func:`os.process_cpu_count`
    2.  :func:`os.sched_getaffinity`
    3.  :func:`os.cpu_count`
    """

    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count()  # type: ignore
    elif hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))  # type: ignore
    else:
        ncpus = os.cpu_count()
        if ncpus is not None:
            return ncpus
        else:
            _log.warning("no CPU count available, assumping single CPU")
            return 1


def _resolve_parallel_config(
    processes: int | list[int] | None = None,
    threads: int | list[int] | None = None,
    backend_threads: int | list[int] | None = None,
    child_threads: int | None = None,
) -> ParallelConfig:
    nprocs = os.environ.get("LK_NUM_PROCS", None)
    nthreads = os.environ.get("LK_NUM_THREADS", None)
    nbthreads = os.environ.get("LK_NUM_BACKEND_THREADS", None)
    cthreads = os.environ.get("LK_NUM_CHILD_THREADS", None)
    ncpus = effective_cpu_count()

    if processes is None and nprocs:
        processes = _parse_numlist(nprocs)

    if threads is None and nthreads:
        threads = _parse_numlist(nthreads)

    if backend_threads is None and nbthreads:
        backend_threads = _parse_numlist(nbthreads)

    if child_threads is None and cthreads:
        warnings.warn(
            "LK_NUM_CHILD_THREADS is deprecated, use list in LK_NUM_BACKEND_THREADS instead",
            DeprecationWarning,
        )
        child_threads = int(cthreads)
    elif child_threads is not None:
        warnings.warn(
            "child_theads is deprecated, use list in backend_threads instead",
            DeprecationWarning,
        )

    if processes is None:
        processes = min(ncpus, 4)

    if threads is None:
        threads = min(ncpus, 8)

    processes = _ensure_numlist(processes)
    threads = _ensure_numlist(threads)

    if backend_threads is None:
        backend_threads = max(min(ncpus // threads[0], 4), 1)

    backend_threads = _ensure_numlist(backend_threads)

    if child_threads is not None:
        backend_threads.append(child_threads)

    return ParallelConfig(processes, threads, backend_threads)


def _unparse_numlist(nums: list[int]) -> str:
    return ",".join([str(i) for i in nums])


def _parse_numlist(spec: str):
    return [int(s.strip()) for s in spec.split(",")]


def _ensure_numlist(nums: int | list[int] | str) -> list[int]:
    if isinstance(nums, str):
        return _parse_numlist(nums)
    elif isinstance(nums, list):
        return nums
    else:
        return [nums]
