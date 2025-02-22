# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import warnings
from typing import Any, Callable, Literal, overload

from ._base import Progress

_backend: Callable[..., Progress] = Progress


@overload
def set_progress_impl(name: Literal["rich"]): ...
@overload
def set_progress_impl(name: Literal["notebook"]): ...
def set_progress_impl(name: str | None, *options: Any):
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
