# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Extended logger providing TRACE support.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import structlog

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
            args, kwargs = self._process_event("trace", event, kw)  # type: ignore
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
