# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from time import sleep

from pytest import importorskip, mark

from lenskit.logging import get_logger
from lenskit.parallel.ray import TaskLimiter, ensure_cluster, ray_available

ray = importorskip("ray")
pytestmark = mark.skipif(not ray_available(), reason="ray is not available")

_log = get_logger(__name__)


@ray.remote
def _dummy_task(n):
    _log.info("worker task %d", n)
    sleep(0.1)
    return n * 100


def test_task_limiter():
    ensure_cluster()
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
    assert any(n >= 2 for n in n_pend)


def test_map():
    ensure_cluster()
    NTASK = 4
    limit = TaskLimiter(NTASK)

    for i, result in enumerate(limit.imap(_dummy_task, range(50))):
        assert result == i * 100


def test_map_unordered():
    ensure_cluster()
    NTASK = 4
    limit = TaskLimiter(NTASK)

    results = set()

    for result in limit.imap(_dummy_task, range(50), ordered=False):
        results.add(result)

    assert results == set(i * 100 for i in range(50))
