"""
Logging, progress, and resource records.
"""

from typing import Any

import structlog

from .config import LoggingConfig
from .progress import Progress, item_progress, set_progress_impl
from .tasks import Task

__all__ = [
    "LoggingConfig",
    "Progress",
    "item_progress",
    "set_progress_impl",
    "Task",
    "get_logger",
    "trace",
]

get_logger = structlog.stdlib.get_logger


def trace(logger: structlog.stdlib.BoundLogger, *args: Any, **kwargs: Any):
    """
    Emit a trace-level message, if LensKit tracing is enabled.  Trace-level
    messages are more fine-grained than debug-level messages, and you usually
    don't want them.
    """
    meth = getattr(logger, "trace", None)
    if meth is not None:
        meth(*args, **kwargs)
