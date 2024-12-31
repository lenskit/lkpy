"""
Logging, progress, and resource records.
"""

import os
from typing import Any

import structlog

from ._proxy import get_logger
from .config import LoggingConfig, basic_logging
from .progress import Progress, item_progress, set_progress_impl
from .tasks import Task

__all__ = [
    "LoggingConfig",
    "basic_logging",
    "Progress",
    "item_progress",
    "set_progress_impl",
    "Task",
    "get_logger",
    "trace",
]

_trace_debug = os.environ.get("LK_TRACE", "no").lower() == "debug"


def trace(logger: structlog.stdlib.BoundLogger, *args: Any, **kwargs: Any):
    """
    Emit a trace-level message, if LensKit tracing is enabled.  Trace-level
    messages are more fine-grained than debug-level messages, and you usually
    don't want them.

    This function does not work on the lazy proxies returned by
    :func:`get_logger` and similar — it only works on bound loggers.

    Stability:
        Caller
    """
    meth = getattr(logger, "trace", None)
    if meth is not None:
        meth(*args, **kwargs)
    elif _trace_debug:
        logger.debug(*args, **kwargs)
