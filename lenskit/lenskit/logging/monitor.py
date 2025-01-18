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
from enum import Enum
from hashlib import blake2b
from tempfile import TemporaryDirectory
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

import zmq

from ._proxy import get_logger
from .tasks import Task

SIGNAL_ADDR = "inproc://lenskit-monitor-signal"
REFRESH_INTERVAL = 5

_log = get_logger(__name__)
_monitor_lock = threading.Lock()
_monitor_instance: Monitor | None = None


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
    global _monitor_instance

    if _monitor_instance is not None:
        return _monitor_instance

    with _monitor_lock:
        if _monitor_instance is None:
            _monitor_instance = Monitor()

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
    """

    zmq: zmq.Context[zmq.Socket[bytes]]
    _backend: MonitorThread
    _signal: zmq.Socket[bytes]
    log_address: str
    _tmpdir: TemporaryDirectory | None = None
    refreshables: dict[UUID, MonitorRefreshable]
    lock: threading.Lock

    def __init__(self):
        self.zmq = zmq.Context()

        addr, log_sock = self._log_sock()
        self.log_address = addr

        self._backend = MonitorThread(self, log_sock)
        self._backend.start()

        self._signal = self.zmq.socket(zmq.REQ)
        self._signal.connect(SIGNAL_ADDR)

        self.lock = threading.Lock()
        self.refreshables = {}

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

    def shutdown(self):
        log = _log.bind()
        try:
            log.debug("requesting monitor shutdown")
            self._signal.send(b"shutdown")
            self._signal.close()

            log.debug("waiting for monitor thread to shut down")
            self._backend.join()

            log.info("monitor shut down")
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
    log_sock: zmq.Socket[bytes]
    poller: zmq.Poller
    _authkey: bytes
    last_refresh: float

    def __init__(self, monitor: Monitor, log_sock: zmq.Socket[bytes]):
        super().__init__(name="LensKitMonitor", daemon=True)
        self.monitor = monitor
        self.log_sock = log_sock
        self._authkey = mp.current_process().authkey
        self.last_refresh = time.perf_counter()

    def run(self) -> None:
        self.state = MonitorState.ACTIVE
        _log.debug("monitor thread started")
        with self.log_sock, self.monitor.zmq.socket(zmq.PULL) as signal:
            signal.bind(SIGNAL_ADDR)
            self.signal = signal

            self.poller = zmq.Poller()
            self.poller.register(signal, zmq.POLLIN)
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

            if self.signal in ready:
                self._handle_signal()

            if self.log_sock in ready:
                try:
                    self._handle_log_message()
                except Exception as e:
                    _log.error("error handling message", exc_info=e)

            if not ready:
                if self.state == MonitorState.DRAINING:
                    self.state = MonitorState.SHUTDOWN

                elif self.state == MonitorState.ACTIVE:
                    # nothing to do — check if we need a refresh
                    now = time.perf_counter()
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
        parts = self.log_sock.recv_multipart()
        if len(parts) != 4:
            _log.warning("invalid multipart message, expected 3 parts")
            return
        engine, name, data, mac = parts

        mb = blake2b(key=self._authkey)
        mb.update(engine)
        mb.update(name)
        mb.update(data)
        if mb.digest() != mac:
            _log.warning("invalid log message digest, dropping")
            return

        engine = engine.decode()
        name = name.decode()

        if engine == "stdlib":
            rec = pickle.loads(data)
            logger = logging.getLogger(name)
            logger.handle(rec)
        elif engine == "structlog":
            logger = get_logger(name)
            data = json.loads(data)
            method = getattr(logger, data["method"])
            method(**data["event"])
        elif engine == "lenskit.logging.tasks":
            task = Task.model_validate_json(data)
            _log.debug("received subtask", task_id=str(task.task_id))
            current = Task.current()
            if current:
                current.add_subtask(task)
            else:
                _log.debug("no active task for subtask reporting")
        else:
            _log.error("invalid log backend")

    def _do_refresh(self):
        with self.monitor.lock:
            objs = list(self.monitor.refreshables.values())

        objs.reverse()

        for obj in objs:
            try:
                obj.monitor_refresh()
            except Exception as e:
                _log.warning("failed to refresh %s: %s", obj, e)
