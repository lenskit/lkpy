# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Utility code for logging and progress reporting from LensKit components.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator
from uuid import uuid4

import torch

from ._base import Progress
from ._dispatch import item_progress

_active_pbs: dict[str, Progress] = {}


@contextmanager
def item_progress_handle(
    *args: item_progress.args, **kwargs: item_progress.kwargs
) -> Generator[str, None, None]:
    with item_progress(*args, **kwargs) as pb:
        h = uuid4().hex
        _active_pbs[h] = pb
        try:
            yield h
        finally:
            del _active_pbs[h]


@torch.jit.ignore  # type: ignore
def pbh_update(h, incr=1):
    # type: (str, int) -> None
    # we need old-school annotations to make pytorch happy
    pb = _active_pbs[h]
    pb.update(incr)
