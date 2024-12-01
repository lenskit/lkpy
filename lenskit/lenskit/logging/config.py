"""
Logging pipeline configuration.
"""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

import structlog

from ._console import ConsoleHandler
from .processors import format_timestamp, remove_internal
from .progress import set_progress_impl

CORE_PROCESSORS = [structlog.processors.add_log_level, structlog.processors.MaybeTimeStamper()]


class LoggingConfig:
    """
    Configuration for LensKit logging.

    This class is intended as a convenience for LensKit applications to set up a
    useful logging and progress reporting configuration; if unconfigured,
    LensKit will emit its logging messages directly to :mod:`structlog`, which
    you can configure in any way you wish.
    """

    level: int = logging.INFO
    file: Path | None = None
    file_level: int | None = None

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
        structlog.configure(
            processors=CORE_PROCESSORS + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                remove_internal,
                format_timestamp,
                structlog.dev.ConsoleRenderer(),
            ],
            foreign_pre_chain=CORE_PROCESSORS,
        )

        level = self.level
        term = ConsoleHandler()
        term.setFormatter(formatter)
        term.setLevel(level)
        root = logging.getLogger()
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

            if file_level < level:
                level = file_level

        root.setLevel(level)

        set_progress_impl("rich")
