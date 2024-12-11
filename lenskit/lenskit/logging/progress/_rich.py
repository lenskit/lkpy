# pyright: strict
from __future__ import annotations

from threading import Lock
from uuid import UUID, uuid4

import structlog
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.progress import Progress as ProgressImpl
from rich.text import Text
from typing_extensions import override

from .._console import console, get_live
from ._base import Progress

_log = structlog.stdlib.get_logger("lenskit.logging.progress")
_pb_lock = Lock()
_progress: ProgressImpl | None = None
_active_bars: dict[UUID, RichProgress] = {}


class RichProgress(Progress):
    """
    Progress bark backed by Rich.
    """

    uuid: UUID
    label: str
    total: int | None
    fields: dict[str, str | None]
    logger: structlog.stdlib.BoundLogger
    _task: TaskID | None = None

    def __init__(self, label: str, total: int | None, fields: dict[str, str | None]):
        super().__init__()
        self.uuid = uuid4()
        self.label = label
        self.total = total
        self.fields = fields

        self.logger = _log.bind(label=label, uuid=str(self.uuid))

        self._task = _install_bar(self)

    def update(self, advance: int = 1, **kwargs: float | int | str):
        if _progress is not None:
            _progress.update(self._task, advance=advance, **kwargs)  # type: ignore

    def finish(self):
        _remove_bar(self)


def _install_bar(bar: RichProgress) -> TaskID | None:
    global _progress
    bar.logger.debug("installing progress bar")
    live = get_live()
    if live is None:
        return None

    with _pb_lock:
        if _progress is None:
            _progress = ProgressImpl(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                RateColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            )
            live.update(_progress)

        _active_bars[bar.uuid] = bar
        return _progress.add_task(bar.label, total=bar.total)


def _remove_bar(bar: RichProgress):
    live = get_live()
    if live is None or _progress is None:
        return
    if bar.uuid not in _active_bars:
        return
    if bar._task is None:
        return

    bar.logger.debug("uninstalling progress bar")

    with _pb_lock:
        _progress.remove_task(bar._task)
        del _active_bars[bar.uuid]


class RateColumn(ProgressColumn):
    def __init__(self):
        super().__init__()

    @override
    def render(self, task: Task):
        speed = task.finished_speed or task.speed
        if speed is None:
            disp = "?"
        elif speed > 1000:
            disp = "{:d} it/s".format(int(speed))
        elif speed > 1:
            disp = "{:.3g} it/s".format(speed)
        else:
            disp = "{:.3g} s/it".format(1.0 / speed)

        return Text(disp, "progress.percentage")
