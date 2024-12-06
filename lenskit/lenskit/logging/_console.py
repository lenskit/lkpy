"""
Console and related logging support
"""

from logging import Handler, LogRecord

from rich.ansi import AnsiDecoder
from rich.console import Console

console = Console(stderr=True)


class ConsoleHandler(Handler):
    """
    Lightweight Rich log handler for routing StructLog-formatted logs.
    """

    _decoder = AnsiDecoder()

    @property
    def supports_color(self) -> bool:
        return console.is_terminal and not console.no_color

    def emit(self, record: LogRecord) -> None:
        try:
            fmt = self.format(record)
            console.print(*self._decoder.decode(fmt))
        except Exception:
            self.handleError(record)
