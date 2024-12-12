"""
Abstraction for recording tasks.
"""

# pyright: strict
from __future__ import annotations

from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

_log = structlog.stdlib.get_logger(__name__)
_active_root: Task | None = None
_active_task: Task | None = None


class Task(BaseModel):
    """
    A task for logging and resource measurement.

    A task may be *top-level* (have no parent), or it may be a *subtask*.  By
    default, new tasks have the current active task as their parent.  Tasks are
    not active until they are started (using a task as a context manager
    automatically does this, which is the recommended process).

    Args:
        label:
            A human-readable label for the task.
    """

    id: UUID = Field(default_factory=uuid4)
    parent: UUID | None = None
    label: str

    def __init__(self, label: str, *, parent: Task | UUID | None = None):
        if isinstance(parent, Task):
            parent = parent.id
        super().__init__(label=label, parent=parent)
