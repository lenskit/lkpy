from __future__ import annotations

from threading import Lock
from uuid import UUID, uuid4

import structlog
from humanize import metric
from rich.console import Group
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.progress import Progress as ProgressImpl
from rich.text import Text

from .._console import console, get_live
from ._base import Progress

_log = structlog.stdlib.get_logger("lenskit.logging.progress")
_pb_lock = Lock()
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

        self._bar = ProgressImpl(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            RateColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        _install_bar(self)

        self._task = self._bar.add_task(label, total=total)

    def update(self, advance: int = 1, **kwargs: float | int | str):
        self._bar.update(self._task, advance=advance, **kwargs)  # type: ignore

    def finish(self):
        self._bar.stop()
        _remove_bar(self)


def _install_bar(bar: RichProgress):
    bar.logger.debug("installing progress bar")
    live = get_live()
    if live is None:
        bar._bar.disable = True
        return

    with _pb_lock:
        _active_bars[bar.uuid] = bar
        rbs = [b._bar for b in _active_bars.values()]
        group = Group(*rbs)
        live.update(group)


def _remove_bar(bar: RichProgress):
    live = get_live()
    if live is None:
        return
    if bar.uuid not in _active_bars:
        return

    bar.logger.debug("uninstalling progress bar")

    with _pb_lock:
        del _active_bars[bar.uuid]

        live.update(Group(*[b._bar for b in _active_bars.values()]))
        live.refresh()


class RateColumn(ProgressColumn):
    def __init__(self):
        super().__init__()

    def render(self, task):
        speed = task.finished_speed or task.speed
        if speed is not None:
            disp = metric(speed, "it/s")
            return Text(disp, "progress.percentage")
        else:
            return Text("?", "progress.percentage")
