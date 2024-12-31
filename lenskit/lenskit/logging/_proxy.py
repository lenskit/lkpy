import logging
from typing import Any

import structlog
from structlog._config import BoundLoggerLazyProxy

_fallback_wrapper = structlog.make_filtering_bound_logger(logging.WARNING)


class LenskitProxyLogger(BoundLoggerLazyProxy):
    """
    Lazy proxy logger for LensKit.  This is based on Structlog's lazy proxy,
    with using a filtering logger by default when structlog is not configured.
    """

    def bind(self, **new_values: Any):
        if structlog.is_configured:
            self._wrapper_class = None
        else:
            self._wrapper_class = _fallback_wrapper

        return super().bind(**new_values)
