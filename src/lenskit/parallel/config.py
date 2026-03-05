# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: basic
from __future__ import annotations

import os
import sys
import sysconfig

import torch
from threadpoolctl import threadpool_limits

from lenskit._accel import init_accel_pool
from lenskit.config import ParallelSettings, lenskit_config
from lenskit.logging import get_logger

__settings: ParallelSettings | None = None
_log = get_logger(__name__)


def get_parallel_config() -> ParallelSettings:
    """
    Get the active parallel configuration, making sure parallelism is configured
    first.
    """
    ensure_parallel_init()
    assert __settings
    return __settings


def init_threading(config: ParallelSettings | None = None):
    """
    Set up and configure LensKit parallelism.  This only needs to be called if
    you want to control when and how parallelism is set up; components using
    parallelism will call :func:`ensure_init`, which will call this function
    with its default arguments if it has not been called.

    .. seealso:: :ref:`parallel-config`
    """
    global __settings
    if __settings:  # pragma: nocover
        _log.warning("parallelism already initialized")
        return

    # our parallel computation doesn't work with FD sharing
    torch.multiprocessing.set_sharing_strategy("file_system")

    if config is None:
        config = lenskit_config().parallel

    __settings = config
    _log.debug("initializing parallelism", config=config)

    nbt = config.num_backend_threads
    if nbt > 0:
        if "OPENBLAS_NUM_THREADS" not in os.environ and "MKL_NUM_THREADS" not in os.environ:
            _log.debug("setting BLAS thread count", num_threads=nbt)
            threadpool_limits(nbt, "blas")

        _log.debug("setting Torch thread count", num_threads=nbt)
        try:
            torch.set_num_threads(nbt)
        except RuntimeError as e:  # pragma: nocover
            _log.warning("failed to configure Pytorch intra-op threads: %s", e)

    assert config.num_threads > 0

    try:
        _log.debug("setting accelerator Rayon thread count", num_threads=config.num_threads)
        init_accel_pool(config.num_threads)
    except RuntimeError as e:  # pragma: nocover
        _log.warning("failed to initialize Rayon backend: %s", e)


def ensure_parallel_init():
    """
    Make sure LensKit parallelism is configured, and configure with defaults if
    it is not.

    Components using parallelism or intensive computations should call this
    function before they begin training.
    """
    if not __settings:
        init_threading()


def is_free_threaded(*, require_active: bool = False) -> bool:
    """
    Query whether this Python supports free-threading.

    Args:
        require_active:
            Require that the GIL is actually disabled (i.e., no modules have
            re-enabled the GIL) in order to return ``True``.
    Returns:
        Whether or not this Python supports free-threading.
    """
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        if require_active and sys._is_gil_enabled():
            return False
        return True
    else:
        return False


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
