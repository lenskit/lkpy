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
from typing import Any, Literal

import structlog
from structlog.stdlib import BoundLogger

from ._proxy import get_logger

__trace_debug = os.environ.get("LK_TRACE", "no").lower() == "debug"
# default based on env var, so that LK_TRACE=debug enabels even without
# debug config — useful for tests.
_tracing_active: bool | Literal["debug"] = __trace_debug


def tracing_active() -> bool:
    """
    Query whether tracing is active.
    """
    return bool(_tracing_active)


def activate_tracing(active: bool | Literal["debug"] = True) -> None:
    """
    Mark tracing as active (or inactive).

    Global tracing state is just used to short-cut tracing.  This method should
    only be called from :class:`~lenskit.logging.LoggingConfig`.

    Args:
        active:
            The global tracing state.  If ``"debug"``, trace messages are
            emitted at DEBUG level.
    """
    global _tracing_active
    _tracing_active = active


def trace(logger: BoundLogger, *args: Any, **kwargs: Any):
    """
    Emit a trace-level message, if LensKit tracing is enabled.  Trace-level
    messages are more fine-grained than debug-level messages, and you usually
    don't want them.

    This function does not work on the lazy proxies returned by
    :func:`get_logger` and similar — it only works on bound loggers.

    Stability:
        Caller
    """
    if _tracing_active == "debug":
        logger.debug(*args, **kwargs)
    elif _tracing_active:
        meth = getattr(logger, "trace", None)
        if meth is not None:
            meth(*args, **kwargs)


def get_tracer(logger: str | BoundLogger, **initial_values: Any):
    """
    Get a tracer for efficient low-level tracing of computations.

    Stability:
        Experimental
    """
    if isinstance(logger, str):
        logger = get_logger(logger)
    if initial_values:
        logger = logger.bind(**initial_values)

    if _tracing_active:
        return ActiveTracer(logger)
    else:
        return Tracer(logger)


class Tracer:
    """
    Logger-like thing that is only for TRACE-level events.

    This class is designed to support structured tracing without the overhead of
    creating and binding new loggers.  It is also imperative, rather than
    functional, so we create fewer objects and so it is a little more ergonomic
    for common tracing flows.

    .. note::

        Don't create instances of this class directly — use
        :func:`~lenskit.logging.get_tracer` to create a tracer.

    Stability:
        Experimental
    """

    _logger: BoundLogger

    def __init__(self, logger: BoundLogger):
        self._logger = logger

    def add_bindings(self, **new_values: Any) -> None:
        """
        Bind new data in the keys.

        .. note::

            Unlike :meth:`structlog.Logger.bind`, this method is **imperative*: it
            updates the tracer in-place instead of returning a new tracer.  If you
            need a new, disconnected tracer, use :meth:`split`.
        """
        pass

    def remove_bindings(self, *keys: str) -> None:
        """
        Unbind keys in the tracer.

        .. note::

            Unlike :meth:`structlog.Logger.bind`, this method is **imperative*: it
            updates the tracer in-place instead of returning a new tracer.  If you
            need a new, disconnected tracer, use :meth:`split`.
        """
        pass

    def reset(self) -> None:
        """
        Reset this tracer's underlying logger to the original logger.
        """
        pass

    def trace(self, event, *args, **bindings):
        """
        Emit a TRACE-level event.
        """
        pass


class ActiveTracer(Tracer):
    """
    Active tracer that actually sends trace messages.
    """

    _base_logger: BoundLogger

    def __init__(self, logger: BoundLogger):
        super().__init__(logger)
        self._base_logger = logger

    def add_bindings(self, **new_values: Any) -> None:
        """
        Bind new data in the keys.

        .. note::

            Unlike :meth:`structlog.Logger.bind`, this method is **imperative*: it
            updates the tracer in-place instead of returning a new tracer.  If you
            need a new, disconnected tracer, use :meth:`split`.
        """
        self._logger = self._logger.bind(**new_values)

    def remove_bindings(self, *keys: str) -> None:
        """
        Unbind keys in the tracer.

        .. note::

            Unlike :meth:`structlog.Logger.bind`, this method is **imperative*: it
            updates the tracer in-place instead of returning a new tracer.  If you
            need a new, disconnected tracer, use :meth:`split`.
        """
        self._logger = self._logger.unbind(*keys)

    def reset(self) -> None:
        """
        Reset this tracer's underlying logger to the original logger.
        """
        self._logger = self._base_logger

    def trace(self, event, *args, **bindings):
        """
        Emit a TRACE-level event.
        """
        trace(self._logger, event, *args, **bindings)


class TracingLogger(structlog.stdlib.BoundLogger):
    """
    Class for LensKit loggers with trace-level logging support.

    Code should not directly use the tracing logger — it should use the
    :func:`trace` function that intelligently checks the logger.
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
