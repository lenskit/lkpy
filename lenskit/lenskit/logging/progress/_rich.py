from __future__ import annotations

from threading import Lock
from uuid import UUID, uuid4

from rich.console import Group
from rich.live import Live
from rich.progress import Progress as ProgressImpl
from rich.progress import TaskID
import structlog

from .._console import console
from ._base import Progress

_log = structlog.stdlib.get_logger('lenskit.logging.progress')
_pb_lock = Lock()
_progress_area: Live | None = None
_active_bars: dict[UUID, RichProgress] = {}


class RichProgress(Progress):
    """
    Progress bark backed by Rich.
    """

    uuid: UUID
    total: int | None
    fields: dict[str, str | None]
    logger: structlog.stdlib.BoundLogger
    _bar: ProgressImpl
    _task: TaskID

    def __init__(self, label: str, total: int | None, fields: dict[str, str | None]):
        super().__init__()
        self.uuid = uuid4()
        self.total = total
        self.fields = fields

        self.logger = _log.bind(label=label, uuid=str(self.uuid))

        self._bar = ProgressImpl(console=console)

        _install_bar(self)

        self._task = self._bar.add_task(label, total=total)

    def update(self, advance: int = 1, **kwargs: float | int | str):
        self._bar.update(self._task, advance=advance, **kwargs)  # type: ignore

    def finish(self):
        self._bar.stop()
        _remove_bar(self)


def _install_bar(bar: RichProgress):
    global _progress_area

    bar.logger.debug('installing progress bar')

    with _pb_lock:
        _active_bars[bar.uuid] = bar
        rbs = [b._bar for b in _active_bars.values()]
        group = Group(*rbs)
        if _progress_area is None:
            _progress_area = Live(group, console=console, transient=True)
            _progress_area.start()
        else:
            _progress_area.update(group)

    _progress_area.refresh()

def _remove_bar(bar: RichProgress):
    global _progress_area
    if bar.uuid not in _active_bars:
        return

    bar.logger.debug('uninstalling progress bar')

    with _pb_lock:
        del _active_bars[bar.uuid]

        assert _progress_area is not None
        _progress_area.update(Group(*[b._bar for b in _active_bars.values()]))
        _progress_area.refresh()
        if not _active_bars:
            _progress_area.stop()
            _progress_area = None
