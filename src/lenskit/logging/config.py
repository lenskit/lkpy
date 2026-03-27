# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Logging pipeline configuration.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Literal, TypeAlias

import structlog
from structlog.dev import RichTracebackFormatter

from ._console import ConsoleHandler, console, setup_console
from .processors import filter_exceptions, format_timestamp, log_warning, remove_internal
from .tracing import activate_tracing, lenskit_filtering_logger

LVL_TRACE = 5
CORE_PROCESSORS = [
    structlog.processors.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.MaybeTimeStamper(),
]
LogFormat: TypeAlias = Literal["json", "logfmt", "text"]

_active_config: LoggingConfig | None = None


def active_logging_config() -> LoggingConfig | None:
    """
    Get the currently-active logging configuration.

    Stability:
        Internal
    """
    return _active_config


def basic_logging(level: int = logging.INFO):
    """
    Simple one-function logging configuration for simple command lines.

    Stability:
        Caller
    """
    cfg = LoggingConfig()
    cfg.level = level
    cfg.apply()


def notebook_logging(level: int = logging.INFO):
    """
    Simple one-function logging configuration for notebooks and similar.

    Stability:
        Caller
    """
    cfg = LoggingConfig()
    cfg.level = level
    cfg.progress_backend = "notebook"
    cfg.apply()


class LoggingConfig:  # pragma: nocover
    """
    Configuration for LensKit logging.

    This class is intended as a convenience for LensKit applications to set up a
    useful logging and progress reporting configuration; if unconfigured,
    LensKit will emit its logging messages directly to :mod:`structlog` and/or
    :mod:`logging`, which you can configure in any way you wish.

    Stability:
        Caller
    """

    level: int = logging.INFO
    stream: Literal["full", "simple", "json"] = "full"
    progress_backend: Literal["notebook", "rich"] | None = None
    file: Path | None = None
    file_level: int | None = None
    file_format: LogFormat = "json"

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

    def set_stream_mode(self, mode: Literal["full", "simple", "json"]):
        """
        Configure the standard error stream mode.
        """
        self.stream = mode
        if mode == "full":
            self.force_console = True

    def set_verbose(self, verbose: bool | int = True):
        """
        Enable verbose logging.

        .. note::

            It is better to only call this method if your application's
            ``verbose`` option is provided, rather than passing your verbose
            option to it, to allow the ``LK_LOG_LEVEL`` environment variable to
            apply in the absence of a configuration option.

        Args:
            verbose:
                The level of verbosity.  Values of ``True`` or ``1`` turn on
                ``DEBUG``-level logs, and ``2`` or greater turns on tracing.
        """
        if isinstance(verbose, int) and verbose > 1:
            self.level = LVL_TRACE
        elif verbose:
            self.level = logging.DEBUG
        else:
            self.level = logging.INFO

    def set_log_file(
        self, path: os.PathLike[str], level: int | None = None, format: LogFormat = "json"
    ):
        """
        Configure a log file.
        """
        self.file = Path(path)
        self.file_level = level
        self.file_format = format

    def apply(self):
        """
        Apply the configuration.
        """
        from .progress import set_progress_impl

        global _active_config

        if self.stream == "full":
            setup_console()

        root = logging.getLogger()

        if self.stream == "json":
            term = logging.StreamHandler(sys.stderr)
            term.setLevel(self.level)
            proc_fmt = structlog.processors.JSONRenderer()
        elif console.is_jupyter:
            term = logging.StreamHandler(sys.stdout)
            term.setLevel(self.level)
            proc_fmt = structlog.dev.ConsoleRenderer(
                colors=self.stream == "full" and not console.no_color,
            )
        else:
            import click

            term = ConsoleHandler()
            term.setLevel(self.level)
            proc_fmt = structlog.dev.ConsoleRenderer(
                colors=self.stream == "full" and term.supports_color,
                exception_formatter=RichTracebackFormatter(
                    show_locals=self.level < logging.INFO, suppress=[click]
                ),
            )

        eff_lvl = self.effective_level
        structlog.configure(
            processors=CORE_PROCESSORS + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
            wrapper_class=lenskit_filtering_logger(eff_lvl),
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                remove_internal,
                format_timestamp,
                filter_exceptions,
                proc_fmt,
            ],
            foreign_pre_chain=CORE_PROCESSORS,
        )

        term.setFormatter(formatter)
        root.addHandler(term)
        if eff_lvl <= LVL_TRACE:
            activate_tracing(True)

        if self.file:
            file_level = self.file_level if self.file_level is not None else self.level
            file = logging.FileHandler(self.file, mode="w")

            if self.file_format == "json":
                proc_fmt = structlog.processors.JSONRenderer()
            elif self.file_format == "logfmt":
                proc_fmt = structlog.processors.LogfmtRenderer(key_order=["event", "timestamp"])
            else:
                proc_fmt = structlog.processors.KeyValueRenderer(key_order=["event", "timestamp"])

            ffmt = structlog.stdlib.ProcessorFormatter(
                processors=[
                    remove_internal,
                    structlog.processors.ExceptionPrettyPrinter(),
                    proc_fmt,
                ],
                foreign_pre_chain=CORE_PROCESSORS,
            )
            file.setFormatter(ffmt)
            file.setLevel(file_level)
            root.addHandler(file)

        root.setLevel(self.effective_level)
        # turn down some loggers
        logging.getLogger("asyncio").setLevel(logging.INFO)
        logging.getLogger("numba").setLevel(logging.INFO)

        if self.progress_backend is not None:
            set_progress_impl(self.progress_backend)
        elif self.stream == "full":
            set_progress_impl("rich")

        warnings.showwarning = log_warning
        warnings.filterwarnings("ignore", message=r"Sparse CSR tensor support is in beta state")

        _active_config = self


def _env_level(name: str) -> int | None:
    ev_level = os.environ.get(name, None)
    if ev_level:
        ev_level = ev_level.strip().upper()
        lmap = logging.getLevelNamesMapping()
        if re.match(r"^\d+$", ev_level):
            return int(ev_level)
        elif ev_level == "TRACE":
            return LVL_TRACE
        elif ev_level in lmap:
            return lmap[ev_level]
        else:
            warnings.warn(f"{name} set to invalid value {ev_level}")
