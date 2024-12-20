"""
Extended logger providing TRACE support.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog


class TracingLogger(structlog.stdlib.BoundLogger):
    """
    Class for LensKit loggers with trace-level logging support.
    """

    def bind(self, **new_values: Any) -> TracingLogger:
        return super().bind(**new_values)  # type: ignore[return-value]

    def new(self, **new_values: Any) -> TracingLogger:
        return super().new(**new_values)  # type: ignore[return-value]

    def trace(self, event: str | None, *args: Any, **kw: Any):
        if args:
            kw["positional_args"] = args
        try:
            args, kwargs = self._process_event("trace", event, kw)
        except structlog.DropEvent:
            return None
        self._logger.debug(*args, **kwargs)


def lenskit_filtering_logger(level: int):
    """
    No-op filtering logger.
    """

    if level < logging.DEBUG:
        return TracingLogger

    def trace(self, event: str | None, *args: Any, **kw: Any):
        pass

    base = structlog.make_filtering_bound_logger(level)
    return type(f"LensKitLoggerFilter{level}", (base,), {"trace": trace})
