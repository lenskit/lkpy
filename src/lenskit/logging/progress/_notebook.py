# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Jupyter notebook progress support.
"""

from __future__ import annotations

from time import perf_counter

import ipywidgets as widgets
from humanize import metric
from IPython.display import display

from .._limit import RateLimit
from ._base import Progress
from ._formats import field_format

__all__ = ["JupyterProgress"]


class JupyterProgress(Progress):  # pragma: nocover
    """
    Progress logging to Jupyter notebook widgets.
    """

    widget: widgets.IntProgress
    text: widgets.Label
    box: widgets.HBox
    current: int
    _limit: RateLimit
    _field_format: str | None = None

    def __init__(
        self,
        label: str | None,
        total: int | None,
        fields: dict[str, str | None],
    ):
        super().__init__()
        self.current = 0
        self.total = total
        if total:
            self.widget = widgets.IntProgress(value=0, min=0, max=total, step=1)
        else:
            self.widget = widgets.IntProgress(value=1, min=0, max=1, step=1)
            self.widget.bar_style = "info"

        pieces = []
        if label:
            pieces.append(widgets.Label(value=label))
        pieces.append(self.widget)

        self.text = widgets.Label()
        if total:
            self.text.value = "0 / {}".format(metric(total))
        pieces.append(self.text)

        self.box = widgets.HBox(pieces)
        display(self.box)

        if fields:
            self._field_format = ", ".join(
                [f"{name}: {field_format(name, fs)}" for (name, fs) in fields.items()]
            )

    def update(
        self,
        advance: int = 1,
        completed: int | None = None,
        total: int | None = None,
        **kwargs: float | int | str,
    ):
        """
        Update the progress bar.
        """
        if total is not None:
            self.total = total
        if completed is not None:
            self.current = completed
        else:
            self.current += advance
        now = perf_counter()
        if self._limit.want_update(now) or (self.total and self.current >= self.total):
            self.widget.value = self.current
            if self.total:
                if self.total >= 1000:
                    txt = "{} / {}".format(metric(self.current), metric(self.total))
                else:
                    txt = "{} / {}".format(self.current, self.total)
            else:
                txt = "{} / ?".format(metric(self.current))
            if self._field_format:
                txt += " ({})".format(self._field_format.format(**kwargs))
            self.text.value = txt

            self._limit.mark_update(now)
        # if self._field_format:
        #     self.tqdm.set_postfix_str(self._field_format.format(kwargs))

    def finish(self):
        """
        Finish and clean up this progress bar.  If the progresss bar is used as
        a context manager, this is automatically called on context exit.
        """
        self.box.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()
