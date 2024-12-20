"""
Logging, progress, and resource records.
"""

from typing import Any

import structlog

from .config import LoggingConfig
from .progress import Progress, item_progress, set_progress_impl
from .tasks import Task
from .tracing import TracingLogger

__all__ = ["LoggingConfig", "Progress", "item_progress", "set_progress_impl", "Task"]


def get_logger(name: str | None = None, **kw: Any) -> TracingLogger:
    """
    Get a logger.  This is a wrapper around :func:`structlog.get_logger` with
    the LensKit tracing logger type annotation.
    """
    return structlog.get_logger(name, **kw)
