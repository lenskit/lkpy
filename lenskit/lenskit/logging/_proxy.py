import logging
from typing import Any

import structlog
from structlog._config import BoundLoggerLazyProxy

_fallback_wrapper = structlog.make_filtering_bound_logger(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger.  This works like :func:`structlog.stdlib.get_logger`, except
    the returned proxy logger is quiet (only WARN and higher messages) if
    structlog has not been configured. LensKit code should use this instead of
    obtaining loggers from Structlog directly.
    """
    return LenskitProxyLogger(None, logger_factory_args=[name])  # type: ignore


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
