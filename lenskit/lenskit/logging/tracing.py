"""
Extended logger providing TRACE support.
"""

import logging
from typing import Any

import structlog


class TracingLogger(structlog.stdlib.BoundLogger):
    """
    Class for LensKit loggers with trace-level logging support.
    """

    def trace(self, event: str | None, *args: Any, **kw: Any):
        self._proxy_to_logger("trace", event, *args, **kw)


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
