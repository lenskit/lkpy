# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from time import sleep

from pytest import importorskip, mark

from lenskit.logging import get_logger
from lenskit.parallel.ray import TaskLimiter, init_cluster, ray_available

ray = importorskip("ray")
pytestmark = mark.skipif(not ray_available(), reason="ray is not available")

_log = get_logger(__name__)


@ray.remote
def _dummy_task(n):
    _log.info("worker task %d", n)
    sleep(0.2)


def test_task_limiter():
    init_cluster()
    NTASK = 4
    limit = TaskLimiter(NTASK)

    n_pend = []

    for i in range(20):
        n_pend.append(limit.pending)
        assert limit.pending <= NTASK
        limit.wait_for_limit()

        task = _dummy_task.remote(i)
        limit.add_task(task)

    assert limit.pending <= NTASK
    limit.drain()
    assert limit.pending == 0

    # make sure we were parallelized
    assert not all(n < 2 for n in n_pend)
