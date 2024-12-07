"""
Support for logging from worker processes.

This is internal support code, clients and LensKit implementers can usually
ignore it.
"""

from __future__ import annotations

import copy
import json
import logging
import multiprocessing as mp
import pickle
from dataclasses import dataclass
from hashlib import blake2b
from logging import Handler, LogRecord, getLogger
from threading import Lock

import structlog
import zmq
from structlog.typing import EventDict

from .config import CORE_PROCESSORS, active_logging_config
from .monitor import get_monitor
from .processors import add_process_info

_active_context: WorkerContext | None = None
_log = structlog.stdlib.get_logger(__name__)


@dataclass
class WorkerLogConfig:
    """
    Configuration for worker logging.
    """

    address: str
    level: int
    authkey: bytes | None = None

    @classmethod
    def current(cls):
        """
        Get the current worker logging configuration.
        """

        if _active_context is not None:
            return _active_context.config
        else:
            mon = get_monitor()
            cfg = active_logging_config()
            level = cfg.effective_level if cfg is not None else logging.INFO
            return cls(address=mon.log_address, level=level)


class WorkerContext:
    """
    Activate (and deactivate) a worker context.  This handles setup and teardown
    of logging, etc.

    Only one worker context can be active, regardless of how many threads are active.
    """

    config: WorkerLogConfig
    zmq: zmq.Context[zmq.Socket[bytes]]
    _log_handler: ZMQLogHandler

    def __init__(self, config: WorkerLogConfig):
        self.config = config
        if self.config.authkey is None:
            self.config.authkey = mp.current_process().authkey

    def start(self):
        global _active_context
        if _active_context is not None:
            raise RuntimeError("worker context already active")

        self.zmq = zmq.Context()
        self._log_handler = ZMQLogHandler(self.zmq, self.config)

        root = getLogger()
        root.addHandler(self._log_handler)
        root.setLevel(self.config.level)

        structlog.configure(
            [add_process_info]
            + CORE_PROCESSORS
            + [structlog.processors.ExceptionPrettyPrinter(), self._log_handler.send_structlog],
            wrapper_class=structlog.make_filtering_bound_logger(self.config.level),
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
        _log.debug("log context activated")

    def shutdown(self):
        root = getLogger()
        root.removeHandler(self._log_handler)

        self._log_handler.shutdown()
        self.zmq.term()

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        self.shutdown()


class ZMQLogHandler(Handler):
    _lock: Lock
    socket: zmq.Socket[bytes]
    key: bytes

    def __init__(self, zmq_context: zmq.Context, config: WorkerLogConfig):
        super().__init__()
        self.config = config
        self._lock = Lock()
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

        key = self.config.authkey
        assert key is not None
        mb = blake2b(key=key)
        engine = b"stdlib"
        mb.update(engine)
        name = record.name.encode()
        mb.update(name)
        data = pickle.dumps(record, pickle.HIGHEST_PROTOCOL)
        mb.update(data)

        with self._lock:
            self.socket.send_multipart([engine, name, data, mb.digest()])

        return record

    def shutdown(self):
        self.socket.close()

    def send_structlog(self, logger, method, event_dict: EventDict):
        key = self.config.authkey
        assert key is not None
        mb = blake2b(key=key)
        engine = b"structlog"
        mb.update(engine)
        name = logger.name.encode()
        mb.update(name)
        data = json.dumps({"method": method, "event": event_dict}).encode()
        mb.update(data)

        with self._lock:
            self.socket.send_multipart([engine, name, data, mb.digest()])

        raise structlog.DropEvent()
