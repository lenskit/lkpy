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
from enum import Enum
from hashlib import blake2b
from tempfile import TemporaryDirectory

import structlog
import zmq

SIGNAL_ADDR = "inproc://lenskit-monitor-signal"

_log = structlog.stdlib.get_logger(__name__)
_monitor_lock = threading.Lock()
_monitor_instance: Monitor | None = None


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

    def __init__(self):
        self.zmq = zmq.Context()

        addr, log_sock = self._log_sock()
        self.log_address = addr

        self._backend = MonitorThread(self, log_sock)
        self._backend.start()

        self._signal = self.zmq.socket(zmq.REQ)
        self._signal.connect(SIGNAL_ADDR)
        _log.bind(address=addr).info("monitor ready")

    def shutdown(self):
        log = _log.bind()
        try:
            log.debug("requesting monitor shutdown")
            self._signal.send_string("shutdown")
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
            self._tmpdir = TemporaryDirectory(prefix="lenskit-monitor", ignore_cleanup_errors=True)
            path = os.path.join(self._tmpdir.name, "log-messages.sock")
            addr = f"ipc://{path}"
            sock.bind(addr)
        else:
            port = sock.bind_to_random_port("127.0.0.1")
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

    def __init__(self, monitor: Monitor, log_sock: zmq.Socket[bytes]):
        super().__init__(name="LensKitMonitor")
        self.monitor = monitor
        self.log_sock = log_sock
        self._authkey = mp.current_process().authkey

    def run(self) -> None:
        self.state = MonitorState.ACTIVE
        _log.debug("monitor thread started")
        with self.log_sock, self.monitor.zmq.socket(zmq.PULL) as signal:
            signal.bind(SIGNAL_ADDR)
            self.signal = signal

            self.poller = zmq.Poller()
            self.poller.register(signal, zmq.POLLIN)
            self.poller.register(signal, zmq.POLLIN)

            while self.state != MonitorState.SHUTDOWN:
                self._pump_message()

            del self.poller

    def _pump_message(self):
        timeout = None
        if self.state == MonitorState.DRAINING:
            timeout = 0

        ready = dict(self.poller.poll(timeout))

        if self.signal in ready:
            self._handle_signal()

        if self.log_sock in ready:
            self._handle_log_message()

        if self.state == MonitorState.DRAINING and not ready:
            self.state = MonitorState.SHUTDOWN

    def _handle_signal(self):
        sig_msg = self.signal.recv_string()
        log = _log.bind(signal=sig_msg)
        match sig_msg:
            case "shutdown":
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
            logger = structlog.get_logger(name)
            data = json.loads(data)
            method = getattr(logger, data["method"])
            method(data["event"], **data["arguments"])
        else:
            _log.error("invalid log backend")
