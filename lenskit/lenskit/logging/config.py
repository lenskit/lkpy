"""
Logging pipeline configuration.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path

import structlog

from ._console import ConsoleHandler, setup_console
from .processors import format_timestamp, log_warning, remove_internal
from .progress import set_progress_impl

CORE_PROCESSORS = [structlog.processors.add_log_level, structlog.processors.MaybeTimeStamper()]

_active_config: LoggingConfig | None = None


def active_logging_config() -> LoggingConfig | None:
    """
    Get the currently-active logging configuration.
    """
    return _active_config


class LoggingConfig:  # pragma: nocover
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

    def __init__(self):
        # initialize configuration from environment variables
        if ev_level := _env_level("LK_LOG_LEVEL"):
            self.level = ev_level

        if ev_file := os.environ.get("LK_LOG_FILE", None):
            self.file = Path(ev_file)

        if ev_level := _env_level("LK_LOG_FILE_LEVEL"):
            self.file_level = ev_level

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

    def log_file(self, path: os.PathLike[str], level: int | None = None):
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


def _env_level(name: str) -> int | None:
    ev_level = os.environ.get(name, None)
    if ev_level:
        ev_level = ev_level.strip().upper()
        lmap = logging.getLevelNamesMapping()
        if re.match(r"^\d+$", ev_level):
            return int(ev_level)
        elif ev_level in lmap:
            return lmap[ev_level]
        else:
            warnings.warn(f"{name} set to invalid value {ev_level}")
