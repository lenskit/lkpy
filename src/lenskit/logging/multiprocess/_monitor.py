# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit background monitoring.

Code almost never needs to interact directly with the monitoring subsystem.
"""

# pyright: basic
from __future__ import annotations

import atexit
import json
import logging
import multiprocessing as mp
import os.path
import pickle
import threading
import time
from contextlib import contextmanager
from enum import Enum
from tempfile import TemporaryDirectory
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

import zmq

from .._proxy import get_logger
from ..tasks import Task
from ._protocol import (
    LogChannel,
    MsgAuthenticator,
    ProgressMessage,
)
from ._records import RecordSink

SIGNAL_ADDR = "inproc://lenskit-monitor-signal"
REFRESH_INTERVAL = 5

_log = get_logger(__name__)
_monitor_lock = threading.Lock()
_monitor_instance: Monitor | None = None


@contextmanager
def maybe_close(sock: zmq.Socket[Any] | None):
    if sock is None:
        yield None
    else:
        try:
            yield sock
        finally:
            sock.close()


@runtime_checkable
class MonitorRefreshable(Protocol):
    def monitor_refresh(self):
        """
        Refresh this object in response to monitor refresh timeouts.
        """
        ...


def get_monitor() -> Monitor:
    """
    Get the monitor, starting it if it is not yet running.
    """
    from ._worker import WorkerLogConfig

    global _monitor_instance

    if _monitor_instance is not None:
        return _monitor_instance

    with _monitor_lock:
        if _monitor_instance is None:
            ctx = WorkerLogConfig.current(from_monitor=False)
            _monitor_instance = Monitor(handle_logging=ctx is None)

        return _monitor_instance


@atexit.register
def _shutdown_monitor():  # type: ignore
    "Shut down the monitor, if one is running."
    global _monitor_instance
    if _monitor_instance is not None:
        _monitor_instance.shutdown()


class MonitorState(Enum):
    SHUTDOWN = 0
    ACTIVE = 1
    DRAINING = 2


class Monitor:
    """
    LensKit monitor controller.

    The monitor does several things:

    * Receive and re-inject log messages from worker processes.
    * Track work in progress and periodically write work logs.

    The monitor is managed and used internally, and neither LensKit client code
    nor component implementations often need to interact with it.

    Args:
        handle_logging:
            Whether or not to handle log messages.
    """

    zmq: zmq.Context[zmq.Socket[bytes]]
    _backend: MonitorThread
    _signal: zmq.Socket[bytes]
    log_address: str | None
    _tmpdir: TemporaryDirectory | None = None
    refreshables: dict[UUID, MonitorRefreshable]
    record_sinks: dict[UUID, RecordSink[Any]]
    lock: threading.Lock

    def __init__(self, handle_logging: bool = True):
        self.zmq = zmq.Context()

        if handle_logging:
            addr, log_sock = self._log_sock()
            self.log_address = addr
        else:
            log_sock = None
            addr = None
            self.log_address = None

        self._backend = MonitorThread(self, log_sock)
        self._backend.start()

        self._signal = self.zmq.socket(zmq.REQ)
        self._signal.connect(SIGNAL_ADDR)

        self.lock = threading.Lock()
        self.refreshables = {}
        self.record_sinks = {}

        _log.bind(address=addr).info("monitor ready")

    def add_refreshable(self, obj: MonitorRefreshable) -> UUID:
        uuid = uuid4()
        with self.lock:
            self.refreshables[uuid] = obj
        return uuid

    def remove_refreshable(self, uuid: UUID):
        with self.lock:
            if uuid in self.refreshables:
                del self.refreshables[uuid]

    def add_record_sink(self, sink: RecordSink[Any]):
        self.record_sinks[sink.sink_id] = sink

    def remove_record_sink(self, sink: RecordSink[Any] | UUID):
        if not isinstance(sink, UUID):
            sink = sink.sink_id

        try:
            del self.record_sinks[sink]
        except KeyError:
            _log.warn("record sink %s already removed", sink)

    def await_quiesce(self, *, ms: int = 100):
        """
        Wait for the monitor to quiesce.

        Args:
            ms:
                The number of milliseconds of quiet to expect for quiescence.
        """
        timeout = ms / 1000
        wait_ns = ms * 1_000_000
        with self._backend.msg_condition:
            while True:
                now = time.perf_counter_ns()
                diff = now - self._backend.last_msg
                if diff >= wait_ns - 1000:
                    return
                _log.debug("last interval %dns, waiting", diff)
                self._backend.msg_condition.wait(timeout)

    def shutdown(self):
        log = _log.bind()
        try:
            log.debug("requesting monitor shutdown")
            self._signal.send(b"shutdown")
            self._signal.close()

            log.debug("waiting for monitor thread to shut down")
            self._backend.join()

            log.debug("monitor shut down")
            self.zmq.term()
        except Exception as e:
            self.zmq.destroy()
            raise e

    def _log_sock(self) -> tuple[str, zmq.Socket[bytes]]:
        sock = self.zmq.socket(zmq.PULL)
        if zmq.has("ipc"):
            self._tmpdir = TemporaryDirectory(prefix="lenskit-monitor.", ignore_cleanup_errors=True)
            path = os.path.join(self._tmpdir.name, "log-messages.sock")
            addr = f"ipc://{path}"
            sock.bind(addr)
        else:
            port = sock.bind_to_random_port("tcp://127.0.0.1")
            addr = f"tcp://127.0.0.1:{port}"
        log = _log.bind(address=addr)
        log.debug("bound listener socket")
        return addr, sock


class MonitorThread(threading.Thread):
    """
    LensKit monitoring backend thread.
    """

    monitor: Monitor
    state: MonitorState

    signal: zmq.Socket[bytes]
    log_sock: zmq.Socket[bytes] | None
    poller: zmq.Poller
    _auth: MsgAuthenticator
    last_refresh: float
    last_msg: int
    msg_condition: threading.Condition

    def __init__(self, monitor: Monitor, log_sock: zmq.Socket[bytes] | None):
        super().__init__(name="LensKitMonitor", daemon=True)
        self.monitor = monitor
        self.log_sock = log_sock
        self._auth = MsgAuthenticator(mp.current_process().authkey)
        self.last_refresh = time.perf_counter()
        self.last_msg = time.perf_counter_ns()
        self.msg_condition = threading.Condition()

    def run(self) -> None:
        self.state = MonitorState.ACTIVE
        _log.debug("monitor thread started")
        with maybe_close(self.log_sock), self.monitor.zmq.socket(zmq.PULL) as signal:
            signal.bind(SIGNAL_ADDR)
            self.signal = signal

            self.poller = zmq.Poller()
            self.poller.register(signal, zmq.POLLIN)
            if self.log_sock is not None:
                self.poller.register(self.log_sock, zmq.POLLIN)

            self._pump_messages()

            del self.poller

    def _pump_messages(self):
        timeout = 0
        last_refresh = time.perf_counter()

        while self.state != MonitorState.SHUTDOWN:
            # don't wait while draining
            if self.state == MonitorState.DRAINING:
                timeout = 0

            ready = dict(self.poller.poll(timeout))
            now_ns = time.perf_counter_ns()
            now = now_ns / 1_000_000_000

            if self.signal in ready:
                self._handle_signal()

            if self.log_sock is not None and self.log_sock in ready:
                try:
                    self._handle_log_message()
                    # can we do this with ZeroMQ instead of a thread lock?
                    with self.msg_condition:
                        self.last_msg = now_ns
                        self.msg_condition.notify_all()
                except Exception as e:
                    _log.error("error handling message", exc_info=e)

            if not ready:
                if self.state == MonitorState.DRAINING:
                    self.state = MonitorState.SHUTDOWN

                elif self.state == MonitorState.ACTIVE:
                    # nothing to do — check if we need a refresh
                    left = max(int((REFRESH_INTERVAL - now + last_refresh) * 1000), 0)
                    if left < 20:
                        self._do_refresh()
                        timeout = REFRESH_INTERVAL * 1000
                    else:
                        timeout = left

    def _handle_signal(self):
        sig_msg = self.signal.recv()
        log = _log.bind(signal=sig_msg)
        match sig_msg:
            case b"":
                log.debug("stray empty message")
            case b"shutdown":
                log.debug("initiating shutdown")
                self.state = MonitorState.DRAINING
            case _:
                log.warning("unknown signal")

    def _handle_log_message(self):
        assert self.log_sock is not None
        parts = self.log_sock.recv_multipart()
        if len(parts) != 4:
            _log.warning("invalid multipart message, expected 3 parts")
            return
        channel, name, data, mac = parts

        if not self._auth.verify_message(channel, name, data, mac):
            _log.warning("invalid log message digest, dropping")
            return

        channel = LogChannel(channel)
        name = name.decode()

        match channel:
            case LogChannel.STDLIB:
                rec = pickle.loads(data)
                logger = logging.getLogger(name)
                logger.handle(rec)
            case LogChannel.STRUCTLOG:
                logger = get_logger(name)
                data = json.loads(data)
                method = getattr(logger, data["method"])
                method(**data["event"])
            case LogChannel.TASKS:
                task = Task.model_validate_json(data)
                _log.debug("received subtask", task_id=str(task.task_id))
                current = Task.current()
                if current:
                    current.add_subtask(task)
                else:
                    _log.debug("no active task for subtask reporting")
            case LogChannel.PROGRESS:
                from ..progress._dispatch import progress_backend

                update = ProgressMessage.model_validate_json(data)
                backend = progress_backend()
                backend.handle_message(update)
            case LogChannel.RECORD:
                sink_id = UUID(name)
                record = pickle.loads(data)
                if sink := self.monitor.record_sinks.get(sink_id):
                    sink.record(record)
                else:
                    _log.warn("received record for nonexistent sink", sink=sink_id.hex)

            case _:
                _log.error("unsupported log channel %s", channel)

    def _do_refresh(self):
        with self.monitor.lock:
            objs = list(self.monitor.refreshables.values())

        objs.reverse()

        for obj in objs:
            try:
                obj.monitor_refresh()
            except Exception as e:
                _log.warning("failed to refresh %s: %s", obj, e)
