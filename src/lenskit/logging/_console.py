# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Console and related logging support
"""

import atexit
import sys
from logging import Handler, LogRecord

from rich.ansi import AnsiDecoder
from rich.console import Console
from rich.live import Live

console = Console(stderr=True)
_live: Live | None = None


class ConsoleHandler(Handler):
    """
    Lightweight Rich log handler for routing StructLog-formatted logs.
    """

    _decoder = AnsiDecoder()

    @property
    def supports_color(self) -> bool:
        return (console.is_terminal or console.is_jupyter) and not console.no_color

    def emit(self, record: LogRecord) -> None:
        try:
            fmt = self.format(record)
            console.print(self._decoder.decode_line(fmt))
        except Exception:
            self.handleError(record)


def get_live() -> Live | None:
    return _live


def setup_console():
    global _live
    if _live is not None:
        return
    if not console.is_terminal:
        return

    _live = Live(console=console, transient=True, redirect_stdout=sys.stdout.isatty())
    _live.start()


def stdout_console():
    """
    Get a console attached to ``stdout``.
    """
    if sys.stdout.isatty():
        return console
    else:
        return Console(stderr=False)


@atexit.register
def _stop_console():
    if _live is not None:
        _live.stop()
