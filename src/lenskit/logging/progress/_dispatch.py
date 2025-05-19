# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

import warnings
from typing import Any, Literal, Protocol, overload

from ..multiprocess._protocol import ProgressMessage
from ._base import Progress

_backend: ProgressBackend = Progress


class ProgressBackend(Protocol):
    """
    Protocol implemented by progress backend objects.
    """

    def __call__(
        self, label: str, total: int | float | None, fields: dict[str, str | None] | None
    ) -> Progress: ...

    @classmethod
    def handle_message(cls, update: ProgressMessage): ...


@overload
def set_progress_impl(name: Literal["rich"]) -> None: ...
@overload
def set_progress_impl(name: Literal["notebook"]) -> None: ...
def set_progress_impl(name: str | None, *options: Any) -> None:
    """
    Set the progress bar implementation.
    """
    global _backend

    match name:
        case "notebook":
            try:
                from ._notebook import JupyterProgress

                _backend = JupyterProgress
            except ImportError:
                warnings.warn("notebook progress backend needs ipywidgets")
                _backend = Progress

        case "rich":
            from ._rich import RichProgress

            _backend = RichProgress

        case "worker":
            from ._worker import WorkerProgress

            _backend = WorkerProgress

        case "none" | None:
            _backend = Progress

        case _:
            raise ValueError(f"unknown progress backend {name}")


def item_progress(
    label: str, total: int | None = None, fields: dict[str, str | None] | None = None
) -> Progress:
    """
    Create a progress bar for distinct, counted items.

    Args:
        label:
            The progress bar label.
        total:
            The total number of items.
        fields:
            Additional fields to report with the progress bar (such as a current
            loss).  These are specified as a dictionary mapping field names to
            format strings (the pieces inside ``{...}`` in :meth:`str.format`),
            and the values come from extra kwargs to :meth:`Progress.update`;
            mapping to ``None`` use default ``str`` formatting.
    """
    return _backend(label, total, fields)


def progress_backend() -> ProgressBackend:
    return _backend
