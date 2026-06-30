# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import threading
from contextvars import Context, copy_context
from dataclasses import dataclass
from typing import Protocol

from lenskit._accel import AtomicInt, NestedAccelPool
from lenskit.logging import Progress, get_logger

from ._pool import NestedPool

_task_count: AtomicInt = AtomicInt()
_log = get_logger(__name__)

UPDATE_INTERVAL = 0.2


def run_accel_task[R](task: AccelTask[R], *, progress: Progress | None = None) -> R:
    """
    Run a accelerated backend task with progress, cancellation, etc.
    """
    thread = AccelTaskThread(task)
    thread.start()

    try:
        while True:
            if progress is not None:
                result = thread.wait_for_result(timeout=UPDATE_INTERVAL)
                cp = task.current_progress()
                if isinstance(cp, tuple):
                    progress.update(completed=cp[0], total=cp[1])
                elif cp is not None:
                    progress.update(completed=cp)
            else:
                result = thread.wait_for_result()

            match result:
                case Exception():
                    raise RuntimeError("accelerator task failed with exception") from result
                case Some(v):
                    return v

    except KeyboardInterrupt as e:
        _log.debug("received KeyboardInterrupt, cancelling background task")
        task.cancel()
        raise e


class AccelTask[R](Protocol):
    """
    An accelerated task, implemented by the accelerator backend.

    This protocol is implemented by long-running accelerator tasks, and is used
    by :func:`run_accel_task` to run these tasks with cancellation, progress
    reporting, and suitable thread pool management.

    The runner will spawn a background thread and call the task's :meth:`invoke`
    method in that thread.  The main thread will poll for completion and
    progress reports.
    """

    def invoke(self, *, pool: NestedAccelPool | None = None) -> R:
        """
        Run this task and return its result.  Must only be called once for a
        given accelerator task.  Will be run on the offloaded execution thread.
        """
        ...

    def cancel(self):
        """
        Cancel this task.

        .. note::

            Will usually **not** be called from the same thread as
            :meth:`invoke`.
        """
        ...

    def current_progress(self) -> int | tuple[int, int] | None:
        """
        Poll the current progress of this task.

        .. note::

            Will usually **not** be called from the same thread as
            :meth:`invoke`.
        """
        ...


@dataclass
class Some[R]:
    value: R


class AccelTaskThread[R](threading.Thread):
    """
    Offloaded execution thread to run an accelerated task.
    """

    _task: AccelTask[R]
    _condition: threading.Condition
    _result: Some[R] | Exception | None = None
    _accel_context: Context

    def __init__(self, task: AccelTask[R]):
        n = _task_count.fetch_add()
        super().__init__(name=f"AccelTask-{n}", daemon=False)
        self._task = task
        self._condition = threading.Condition()
        self._accel_context = copy_context()

    def run(self):
        try:
            _log.debug("beginning accelerator task", task=self._task)
            res = self._accel_context.run(self._task.invoke, pool=NestedPool.active_accel_pool())
            res = Some(res)
            _log.debug("accelerator task finished", task=self._task)
        except Exception as e:
            _log.error("accelerator task failed", task=self._task, exc_info=e)
            res = e

        self._set_result(res)

    def _set_result(self, result: Some[R] | Exception):
        with self._condition:
            self._result = result
            self._condition.notify_all()

    def wait_for_result(self, *, timeout: float | None = None) -> Some[R] | Exception | None:
        with self._condition:
            if self._result is not None:
                return self._result
            self._condition.wait(timeout=timeout)
            return self._result
