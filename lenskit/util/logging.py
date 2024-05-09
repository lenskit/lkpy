# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
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
from progress_api import Progress, make_progress

_active_pbs: dict[str, Progress] = {}


@contextmanager
def progress_handle(
    *args: make_progress.args, **kwargs: make_progress.kwargs
) -> Generator[str, None, None]:
    with make_progress(*args, **kwargs) as pb:
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
