# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import warnings
from contextvars import ContextVar, Token

from lenskit._accel import NestedAccelPool

from .config import get_parallel_config

_active_pool: ContextVar[NestedAccelPool | None] = ContextVar("lenskit:active_pool", default=None)


class NestedPool:
    """
    Context manager to run accelerator tasks in separate accelerator pools.

    Stability:
        Internal
    """

    n_threads: int
    _pool: NestedAccelPool | None = None
    _token: Token | None = None

    def __init__(self, *, n_threads: int | None = None):
        if n_threads is None:
            n_threads = get_parallel_config().num_threads

        self.n_threads = n_threads

    @staticmethod
    def active_accel_pool() -> NestedAccelPool | None:
        return _active_pool.get()

    def __enter__(self):
        assert self._pool is None
        self._pool = NestedAccelPool(self.n_threads)
        self._token = _active_pool.set(self._pool)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        else:
            warnings.warn("nested pool already shut down")
        if self._token is not None:
            _active_pool.reset(self._token)
            self._token = None
