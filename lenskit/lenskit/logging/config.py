"""
Logging pipeline configuration.
"""

from __future__ import annotations

import logging
import warnings
from os import PathLike
from pathlib import Path

import structlog

from ._console import ConsoleHandler, setup_console
from .processors import format_timestamp, remove_internal
from .progress import set_progress_impl

CORE_PROCESSORS = [structlog.processors.add_log_level, structlog.processors.MaybeTimeStamper()]

_active_config: LoggingConfig | None = None


def active_logging_config() -> LoggingConfig | None:
    """
    Get the currently-active logging configuration.
    """
    return _active_config


class LoggingConfig:
    """
    Configuration for LensKit logging.

    This class is intended as a convenience for LensKit applications to set up a
    useful logging and progress reporting configuration; if unconfigured,
    LensKit will emit its logging messages directly to :mod:`structlog` and/or
    :mod:`logging`, which you can configure in any way you wish.
    """

    level: int = logging.INFO
    file: Path | None = None
    file_level: int | None = None

    @property
    def effective_level(self) -> int:
        if self.file_level is not None and self.file_level < self.level:
            return self.file_level
        else:
            return self.level

    def set_verbose(self, verbose: bool = True):
        """
        Enable verbose logging.
        """
        if verbose:
            self.level = logging.DEBUG
        else:
            self.level = logging.INFO

    def log_file(self, path: PathLike[str], level: int | None = None):
        """
        Configure a log file.
        """
        self.file = Path(path)
        self.file_level = level

    def apply(self):
        """
        Apply the configuration.
        """
        global _active_config

        setup_console()
        root = logging.getLogger()
        term = ConsoleHandler()
        term.setLevel(self.level)

        structlog.configure(
            processors=CORE_PROCESSORS + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
            wrapper_class=structlog.make_filtering_bound_logger(self.effective_level),
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                remove_internal,
                format_timestamp,
                structlog.dev.ConsoleRenderer(colors=term.supports_color),
            ],
            foreign_pre_chain=CORE_PROCESSORS,
        )

        term.setFormatter(formatter)
        root.addHandler(term)

        if self.file:
            file_level = self.file_level if self.file_level is not None else self.level
            file = logging.FileHandler(self.file, mode="w")
            ffmt = structlog.stdlib.ProcessorFormatter(
                processors=[
                    remove_internal,
                    structlog.processors.ExceptionPrettyPrinter(),
                    structlog.processors.JSONRenderer(),
                ],
                foreign_pre_chain=CORE_PROCESSORS,
            )
            file.setFormatter(ffmt)
            file.setLevel(file_level)
            root.addHandler(file)

        root.setLevel(self.effective_level)

        set_progress_impl("rich")
        warnings.showwarning = log_warning

        _active_config = self


def log_warning(message, category, filename, lineno, file=None, line=None):
    log = structlog.stdlib.get_logger()
    log.bind(file=filename, lineno=line, category=category.__name__)
    log.warning(f"{category.__name__}: {message}")
