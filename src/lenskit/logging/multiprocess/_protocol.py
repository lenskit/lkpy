# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from enum import Enum
from uuid import UUID

from pydantic import BaseModel


class LogChannel(Enum):
    STDLIB = b"stdlib"
    STRUCTLOG = b"structlog"
    TASKS = b"lenskit.logging.tasks"
    PROGRESS = b"lenskit.logging.progress"


class ProgressMessage(BaseModel):
    progress_id: UUID
    label: str
    total: int | float | None
    completed: int | float | None
    fields: dict[str, int | float | str]
    field_formats: dict[str, str | None]
    finished: bool = False
