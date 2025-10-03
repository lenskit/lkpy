# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

from threading import RLock
from uuid import UUID

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
from typing_extensions import TYPE_CHECKING, override

from .._console import console, get_live
from .._proxy import get_logger
from ._base import Progress
from ._formats import field_format

if TYPE_CHECKING:
    from ..multiprocess._protocol import ProgressMessage

_log = get_logger("lenskit.logging.progress")
_pb_lock = RLock()
_progress: ProgressImpl | None = None
_active_bars: dict[UUID, RichProgress] = {}


class RichProgress(Progress):
    """
    Progress bark backed by Rich.
    """

    label: str
    logger: structlog.stdlib.BoundLogger
    _field_format: str | None = None
    _task: TaskID | None = None

    def __init__(
        self,
        label: str,
        total: int | float | None,
        fields: dict[str, str | None] | None,
        *,
        uuid: UUID | None = None,
    ):
        super().__init__(uuid=uuid)

        self.label = label
        self.total = total

        self.logger = _log.bind(label=label, uuid=str(self.uuid))

        self._task = self._install()

        if fields:
            self._field_format = _make_format(fields)

    @classmethod
    def handle_message(cls, update: ProgressMessage):
        with _pb_lock:
            pb = _active_bars.get(update.progress_id)
            if pb is None:
                pb = cls(update.label, update.total, {}, uuid=update.progress_id)

        if update.finished:
            pb.finish()
            return

        fields = {}
        if update.fields:
            cls._field_format = _make_format(
                {n: f.format for (n, f) in update.fields.items() if f.value is not None}
            )
            fields = {n: f.value for (n, f) in update.fields.items() if f.value is not None}
        else:
            cls._field_format = ""

        pb.update(advance=0, completed=update.completed, total=update.total, **fields)

    def update(
        self,
        advance: int = 1,
        completed: int | float | None = None,
        total: int | float | None = None,
        **kwargs: float | int | str,
    ):
        extra = ""
        if self._field_format:
            extra = self._field_format.format(**kwargs)
        if total is not None:
            self.total = total
        if _progress is not None:
            _progress.update(
                self._task,  # type: ignore
                advance=advance if completed is None else None,
                completed=completed,
                total=total,
                extra=extra,
            )

    def finish(self):
        self._remove()

    def _install(self) -> TaskID | None:
        global _progress
        self.logger.debug("installing progress bar")
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
                    TextColumn("{task.fields[extra]}"),
                    console=console,
                )
                live.update(_progress)

            _active_bars[self.uuid] = self
            return _progress.add_task(self.label, total=self.total, extra="")

    def _remove(self):
        live = get_live()
        if live is None or _progress is None:
            return
        if self.uuid not in _active_bars:
            return
        if self._task is None:
            return

        self.logger.debug("uninstalling progress bar")

        with _pb_lock:
            _progress.remove_task(self._task)
            del _active_bars[self.uuid]


def _make_format(fields: dict[str, str | None]) -> str:
    return ", ".join(
        [f"[json.key]{name}[/json.key]: {field_format(name, fs)}" for (name, fs) in fields.items()]
    )


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
