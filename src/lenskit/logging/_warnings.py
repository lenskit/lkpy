from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Any

from structlog import DropEvent
from structlog.types import EventDict

from ._proxy import get_logger

_batch_lock = Lock()
_warning_batch: ContextVar[dict[str, WarningBatch] | None] = ContextVar(
    "lenskit.logging.warning_batch",
    default=None,
)


@dataclass
class WarningBatch:
    logger: str
    message: str
    first_time: float
    last_time: float = 0
    count: int = 0


def collect_warnings(logger: Any, method: str, event_dict: EventDict) -> EventDict:
    """
    Structlog processor that collects warnings when inside :func:`batch_warnings`.
    """
    if method != "warning":
        return event_dict

    collector = _warning_batch.get()
    if collector is None:
        return event_dict

    if key := event_dict.get("_batch"):
        with _batch_lock:
            batch = collector.setdefault(
                key,
                WarningBatch(logger.name, event_dict["event"], first_time=event_dict["timestamp"]),
            )
            batch.count += 1
            batch.last_time = event_dict["timestamp"]
            if batch.count > 1:
                raise DropEvent()

    return event_dict


@contextmanager
def batch_warnings():
    """
    Collect repeated warnings, and emit them once (with a count) at the end of the job.

    This function is a context manager::

        with batch_warnings():
            # work to do with collected warnings
    """

    token = _warning_batch.set({})
    try:
        yield
    finally:
        counts = _warning_batch.get()
        assert counts is not None
        _warning_batch.reset(token)
        _log_warning_batches(counts)


def _log_warning_batches(batches: Mapping[str, WarningBatch]):
    """
    Print out the logged batches of warnings.
    """

    # preserve first-time ordering for warnings
    keys = sorted(batches.keys(), key=lambda k: batches[k].first_time)
    for key in keys:
        batch = batches[key]

        # we printed warning on first (and only) use
        if batch.count == 1:
            continue

        n = batch.count - 1
        log = get_logger(batch.logger)
        log.warning("warning repeated %d more times: %s", n, batch.message, count=n)
