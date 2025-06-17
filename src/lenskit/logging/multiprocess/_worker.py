# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Support for logging from worker processes.

This is internal support code, clients and LensKit implementers can usually
ignore it.
"""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import pickle
import warnings
from dataclasses import dataclass
from logging import Handler, LogRecord, getLogger
from threading import Lock
from time import perf_counter
from typing import Self, overload

import structlog
import zmq
from structlog.typing import EventDict

from lenskit.logging.progress._base import Progress

from .._limit import RateLimit
from .._proxy import get_logger
from ..config import CORE_PROCESSORS, active_logging_config, log_warning
from ..processors import add_process_info
from ..progress import set_progress_impl

# from ..progress._worker import ProgressMessage
from ..tasks import Task
from ..tracing import lenskit_filtering_logger
from ._protocol import (
    LogChannel,
    MsgAuthenticator,
    ProgressField,
    ProgressMessage,
)

_active_context: WorkerContext | None = None
_log = get_logger(__name__)


@dataclass
class WorkerLogConfig:
    """
    Configuration for worker logging.
    """

    address: str
    level: int
    authkey: bytes | None = None

    @classmethod
    @overload
    def current(cls) -> Self: ...
    @classmethod
    @overload
    def current(cls, *, from_monitor: bool = True) -> Self | None: ...
    @classmethod
    def current(cls, *, from_monitor: bool = True):
        """
        Get the current worker logging configuration.
        """

        if _active_context is not None:
            return _active_context.config
        elif from_monitor:
            from ..monitor import get_monitor

            mon = get_monitor()
            if mon.log_address is None:
                raise RuntimeError("monitor has no log address")
            cfg = active_logging_config()
            level = cfg.effective_level if cfg is not None else logging.INFO
            return cls(
                address=mon.log_address, level=level, authkey=bytes(mp.current_process().authkey)
            )
        else:
            return None


class WorkerContext:
    """
    Activate (and deactivate) a worker context.  This handles setup and teardown
    of logging, etc.

    Only one worker context can be active, regardless of how many threads are active.

    Stability:
        internal
    """

    config: WorkerLogConfig
    zmq: zmq.Context[zmq.Socket[bytes]]
    _log_handler: ZMQLogHandler
    _ref_count: int = 0

    def __init__(self, config: WorkerLogConfig):
        self.config = config
        if self.config.authkey is None:
            self.config.authkey = mp.current_process().authkey

    @staticmethod
    def active() -> WorkerContext | None:
        return _active_context

    def start(self):
        """
        Start the logging context.
        """
        global _active_context
        if _active_context is not None:
            raise RuntimeError("worker context already active")
        _active_context = self

        self.zmq = zmq.Context()
        self._log_handler = ZMQLogHandler(self.zmq, self.config)

        root = getLogger()
        root.addHandler(self._log_handler)
        root.setLevel(self.config.level)

        structlog.configure(
            [add_process_info]
            + CORE_PROCESSORS
            + [structlog.processors.ExceptionPrettyPrinter(), self._log_handler.send_structlog],
            wrapper_class=lenskit_filtering_logger(self.config.level),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,
        )
        warnings.showwarning = log_warning
        set_progress_impl(WorkerProgress)
        _log.debug("log context activated")

    def shutdown(self):
        global _active_context
        root = getLogger()
        root.removeHandler(self._log_handler)
        set_progress_impl(None)

        self._log_handler.shutdown()
        self.zmq.term()
        _active_context = None

    def send_task(self, task: Task):
        self._log_handler.send_task(task)

    def send_progress(self, update: ProgressMessage):
        """
        Send a progrss update event.
        """
        self._log_handler.send_progress(update)

    def __enter__(self):
        if self._ref_count == 0:
            self.start()
        self._ref_count += 1
        return self

    def __exit__(self, *args):
        self._ref_count -= 1
        if self._ref_count == 0:
            self.shutdown()


class WorkerProgress(Progress):  # pragma: nocover
    """
    Progress logging over the pipe to a supervisor.
    """

    label: str
    context: WorkerContext | None
    completed: float = 0
    _fields: dict[str, str | None] = {}
    _limit: RateLimit

    def __init__(
        self,
        label: str,
        total: int | None,
        fields: dict[str, str | None],
    ):
        super().__init__()
        self.context = WorkerContext.active()
        self.label = label
        self.total = total
        self._fields = fields
        self._limit = RateLimit(20)

    def update(
        self,
        advance: int = 1,
        completed: int | None = None,
        total: int | None = None,
        **kwargs: float | int | str,
    ):
        if self.context is None:
            return

        if completed is not None:
            self.completed = completed
        else:
            self.completed += advance
        if total is not None:
            self.total = total

        now = perf_counter()
        if self._limit.want_update(now) or self.completed == self.total:
            fields = {
                name: ProgressField(value, self._fields[name])
                for (name, value) in kwargs.items()
                if name in kwargs
            }
            self.context.send_progress(
                ProgressMessage(
                    progress_id=self.uuid,
                    label=self.label,
                    total=self.total,
                    completed=self.completed,
                    fields=fields,
                )
            )
            self._limit.mark_update(now)

    def finish(self):
        if self.context is None:
            return

        self.context.send_progress(
            ProgressMessage(
                progress_id=self.uuid,
                label=self.label,
                total=self.total,
                completed=self.completed,
                finished=True,
            )
        )


class ZMQLogHandler(Handler):
    _lock: Lock
    socket: zmq.Socket[bytes]
    key: bytes
    _render = structlog.processors.JSONRenderer()
    _auth: MsgAuthenticator

    def __init__(self, zmq_context: zmq.Context, config: WorkerLogConfig):
        super().__init__()
        self.config = config
        self._lock = Lock()
        assert config.authkey is not None
        self._auth = MsgAuthenticator(config.authkey)
        self.socket = zmq_context.socket(zmq.PUSH)
        self.socket.connect(config.address)

    def handle(self, record: LogRecord) -> LogRecord | bool:  # type: ignore
        # copy so other handlers don't have a problem
        record = copy.copy(record)

        # update messages for copyability
        if not hasattr(record, "message"):
            record.message = record.msg % record.args

        record.exc_info = None
        record.exc_text = None
        record.stack_info = None

        self._send_message(
            LogChannel.STDLIB, record.name.encode(), pickle.dumps(record, pickle.HIGHEST_PROTOCOL)
        )

        return record

    def shutdown(self):
        self.socket.close()

    def send_structlog(self, logger, method, event_dict: EventDict):
        x = self._render(logger, method, {"method": method, "event": event_dict})
        if isinstance(x, str):
            x = x.encode()
        self._send_message(LogChannel.STRUCTLOG, logger.name.encode(), x)

        raise structlog.DropEvent()

    def send_task(self, task: Task):
        _log.debug("sending updated task", task_id=task.task_id)
        self._send_message(
            LogChannel.TASKS, str(task.task_id).encode(), task.model_dump_json().encode()
        )

    def send_progress(self, update: ProgressMessage):
        self._send_message(
            LogChannel.PROGRESS,
            str(update.progress_id).encode(),
            update.model_dump_json().encode(),
        )

    def _send_message(self, channel: LogChannel, name: bytes, data: bytes):
        mac = self._auth.hash_message(channel, name, data)

        with self._lock:
            self.socket.send_multipart([channel.value, name, data, mac])


def send_task(task: Task):
    assert _active_context is not None
    _active_context.send_task(task)
