# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload, runtime_checkable

from ._base import Progress

if TYPE_CHECKING:
    from ..multiprocess._protocol import ProgressMessage

_backend: ProgressBackend = Progress


@runtime_checkable
class ProgressBackend(Protocol):
    """
    Protocol implemented by progress backend objects.
    """

    def __call__(
        self, label: str, total: int | float | None, fields: Mapping[str, str | None] | None
    ) -> Progress: ...

    @classmethod
    def handle_message(cls, update: ProgressMessage): ...


@overload
def set_progress_impl(name: Literal["rich", "notebook", "none"] | None) -> None: ...
@overload
def set_progress_impl(backend: ProgressBackend, /) -> None: ...
def set_progress_impl(name: str | ProgressBackend | None, *options: Any) -> None:
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

        case "none" | None:
            _backend = Progress

        case ProgressBackend():
            _backend = name

        case _:
            raise ValueError(f"unknown progress backend {name}")


def item_progress(
    label: str, total: int | None = None, fields: Mapping[str, str | None] | None = None
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
