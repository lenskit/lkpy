# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from enum import Enum
from hashlib import blake2b
from typing import NamedTuple
from uuid import UUID

from pydantic import BaseModel


class MsgAuthenticator:
    _authkey: bytes

    def __init__(self, key: bytes):
        self._authkey = key

    def hash_message(self, channel: LogChannel | bytes, name: bytes, data: bytes) -> bytes:
        mb = blake2b(key=self._authkey)
        if isinstance(channel, bytes):
            mb.update(channel)
        else:
            mb.update(channel.value)
        mb.update(name)
        mb.update(data)
        return mb.digest()

    def verify_message(
        self, channel: LogChannel | bytes, name: bytes, data: bytes, hash: bytes
    ) -> bool:
        return self.hash_message(channel, name, data) == hash


class LogChannel(Enum):
    STDLIB = b"stdlib"
    STRUCTLOG = b"structlog"
    TASKS = b"lenskit.logging.tasks"
    PROGRESS = b"lenskit.logging.progress"


class ProgressField(NamedTuple):
    value: int | float | str
    format: str | None = None


class ProgressMessage(BaseModel):
    progress_id: UUID
    label: str
    total: int | float | None
    completed: int | float | None
    fields: dict[str, ProgressField] = {}
    finished: bool = False
