# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Subprocess progress support.
"""

from __future__ import annotations

from time import perf_counter

from .._limit import RateLimit
from ..multiprocess._protocol import ProgressMessage
from ..worker import WorkerContext
from ._base import Progress

__all__ = ["WorkerProgress"]


class WorkerProgress(Progress):  # pragma: nocover
    """
    Progress logging over the pipe to a supervisor.
    """

    label: str
    context: WorkerContext
    completed: float = 0
    _fields: dict[str, str | None] = {}
    _limit: RateLimit

    def __init__(
        self,
        context: WorkerContext,
        label: str,
        total: int | None,
        fields: dict[str, str | None],
    ):
        super().__init__()
        self.context = context
        self.label = label
        self.total = total
        self._fields = fields
        self._limit = RateLimit(20)

    def update(
        self,
        advance: int = 1,
        completed: int | None = None,
        total: int | None = None,
        **kwargs: float | int | str,
    ):
        if completed is not None:
            self.completed = completed
        else:
            self.completed += advance
        if total is not None:
            self.total = total

        now = perf_counter()
        if self._limit.want_update(now):
            self.context.send_progress(
                ProgressMessage(
                    progress_id=self.uuid,
                    label=self.label,
                    total=self.total,
                    completed=self.completed,
                    fields=kwargs,
                    field_formats=self._fields,
                )
            )

    def finish(self):
        self.context.send_progress(
            ProgressMessage(
                progress_id=self.uuid,
                label=self.label,
                total=self.total,
                completed=self.completed,
                finished=True,
            )
        )
