import logging
import re
from typing import Any

import structlog
from structlog._config import BoundLoggerLazyProxy

_fallback_wrapper = structlog.make_filtering_bound_logger(logging.WARNING)


def get_logger(
    name: str, *, remove_private: bool = True, **init_als: Any
) -> structlog.stdlib.BoundLogger:
    """
    Get a logger.  This works like :func:`structlog.stdlib.get_logger`, except
    the returned proxy logger is quiet (only WARN and higher messages) if
    structlog has not been configured. LensKit code should use this instead of
    obtaining loggers from Structlog directly.

    It also suppresses private module name components of the logger name, so
    e.g. ``lenskit.pipeline._impl`` becomes ``lenskit.pipeline`.

    Params:
        name:
            The logger name.
        remove_private:
            Set to ``False`` to keep private module components of the logger
            name instead of removing them.
        init_vals:
            Initial values to bind into the logger when crated.
    Returns:
        A lazy proxy logger.  The returned logger is type-compatible with
        :class:`structlib.stdlib.BoundLogger`, but is actually an instance of an
        internal proxy that provies more sensible defaults and handles LensKit's
        TRACE-level logging support.
    """
    if remove_private:
        name = re.sub(r"\._.*", "", name)
    return LenskitProxyLogger(None, logger_factory_args=[name], initial_values=init_als)  # type: ignore


class LenskitProxyLogger(BoundLoggerLazyProxy):
    """
    Lazy proxy logger for LensKit.  This is based on Structlog's lazy proxy,
    with using a filtering logger by default when structlog is not configured.
    """

    def bind(self, **new_values: Any):
        if structlog.is_configured():
            self._wrapper_class = None
        else:
            self._wrapper_class = _fallback_wrapper

        return super().bind(**new_values)
