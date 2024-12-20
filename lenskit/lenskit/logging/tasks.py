"""
Abstraction for recording tasks.
"""

# pyright: strict
from __future__ import annotations

from enum import Enum
from os import PathLike
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from .resource import ResourceMeasurement, reset_linux_hwm

_log = structlog.stdlib.get_logger(__name__)
_active_tasks: list[Task] = []


class TaskStatus(str, Enum):
    """
    Statuses for task records.
    """

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel, extra="allow"):
    """
    A task for logging and resource measurement.

    A task may be *top-level* (have no parent), or it may be a *subtask*.  By
    default, new tasks have the current active task as their parent.  Tasks are
    not active until they are started (using a task as a context manager
    automatically does this, which is the recommended process).

    The task-tracking mechanism is currently designed to support large tasks,
    like training a model or running batch inference; it is not yet designed for
    fine-grained spans like you would see in OpenTelemetry or Eliot.

    .. note::

        The notion of the “active task” does not yet support multi-threaded
        tasks.

    Args:
        file:
            A file to save the task when it is finished.
        parent:
            The parent task.  If unspecified, uses the currently-active task.
        reset_hwm:
            Whether to reset the system resource high-water-marks at the start
            of this task.  Only effective on Linux, but allows for measurement
            of the peak memory use of this task specifically.  If unspecified,
            it resets the HWM if there is no parent.
    """

    task_id: UUID = Field(default_factory=uuid4, frozen=True)
    """
    The task ID.
    """
    parent_id: UUID | None = Field(default=None, frozen=True)
    """
    The parent task ID.
    """
    subprocess: bool = False
    """
    Whether this task is a subprocess of its parent.  Subprocess task CPU times
    are *not* included in the parent task times.
    """
    label: str
    """
    Human-readable task label.
    """

    status: TaskStatus = TaskStatus.PENDING
    "The task's current status."
    start_time: float | None = None
    """
    The task start time (UNIX timestamp).
    """
    finish_time: float | None = None
    """
    The task completion time (UNIX timestamp).
    """

    duration: float | None = None
    """
    Task duration in seconds.  Measured using :func:`time.perf_counter`, so it
    may disagree slightly with the difference in start and finish times.
    """
    cpu_time: float | None = None
    """
    CPU time consumed in seconds.
    """
    peak_memory: int | None = None
    """
    Peak memory usage (max RSS) in bytes.  Only available on Unix; individual
    task peak memory use is only reliable on Linux (MacOS will report the max
    memory used since the process was started).
    """
    peak_gpu_memory: int | None = None
    """
    Peak PyTorch GPU memory usage in bytes.
    """

    subtasks: dict[UUID, Task] = Field(default_factory=dict)
    """
    This task's subtasks.
    """

    _reset_hwm: bool = False
    _save_file: Path | None = None
    _lock: Lock = None  # type: ignore
    _initial_meter: ResourceMeasurement | None = None
    _final_meter: ResourceMeasurement | None = None
    _refresh_id: UUID | None = None

    @staticmethod
    def current() -> Task | None:
        """
        Get the currently-active task.
        """
        if _active_tasks:
            return _active_tasks[-1]

    @staticmethod
    def root() -> Task | None:
        """
        Get the root task.
        """
        if _active_tasks:
            return _active_tasks[0]

    def __init__(
        self,
        label: str,
        *,
        file: PathLike[str] | None = None,
        parent: Task | UUID | None = None,
        reset_hwm: bool | None = None,
        **data: Any,
    ):
        if isinstance(parent, Task):
            data["parent_id"] = parent.task_id
        elif parent is None and _active_tasks:
            data["parent_id"] = _active_tasks[-1].task_id
        super().__init__(label=label, **data)
        if reset_hwm is not None:
            self._reset_hwm = reset_hwm
        else:
            self._reset_hwm = parent is not None

        self._lock = Lock()

        if file:
            self._save_file = Path(file)

    def save_to_file(self, path: PathLike[str], monitor: bool = True):
        """
        Save this task to a file, and re-save it when finished.
        """
        self._save_file = Path(path)
        self._save()

        # add to monitor if it isn't already
        if (
            monitor
            and self._initial_meter is not None
            and self._refresh_id is None
            and self._final_meter is None
        ):
            from lenskit.logging.monitor import get_monitor

            mon = get_monitor()
            self._refresh_id = mon.add_refreshable(self)

    def start(self):
        """
        Start the task.
        """
        if self._reset_hwm:
            reset_linux_hwm()

        log = _log.bind(task_id=self.task_id)
        self._initial_meter = ResourceMeasurement.current()
        self.start_time = self._initial_meter.wall_time
        self.status = TaskStatus.RUNNING
        log.debug("beginning task")
        self._save()

        if self.parent_id:
            cur = self.current()
            if cur is not None:
                if cur.task_id == self.parent_id:
                    cur.add_subtask(self)
                else:
                    log.warn("active task is not parent")
            else:
                log.debug("have a task but no parent")

        _active_tasks.append(self)
        if self._save_file:
            from lenskit.logging.monitor import get_monitor

            mon = get_monitor()
            self._refresh_id = mon.add_refreshable(self)

    def finish(self, status: TaskStatus = TaskStatus.FINISHED):
        """
        Finish the task.
        """
        log = _log.bind(task_id=str(self.task_id), label=self.label)
        if _active_tasks[-1] is not self:
            log.warn("attempted to finish non-active task")
            _active_tasks.remove(self)
        else:
            _active_tasks.pop()

        with self._lock:
            if self._refresh_id is not None:
                from lenskit.logging.monitor import get_monitor

                mon = get_monitor()
                mon.remove_refreshable(self._refresh_id)

            self._final_meter = self.update_resources()
            self.finish_time = self._final_meter.wall_time
            self.status = status
            log.debug("finished task", time=self.duration, cpu=self.cpu_time)
            self._save()

    def update(self):
        """
        Update the task's resource measurements and save the file (if one is set).
        """
        with self._lock:
            self.update_resources()
            self._save()

    def monitor_refresh(self):
        """
        Refresh method called by the monitor backend.
        """
        self.update()

    def add_subtask(self, task: Task):
        """
        Add or update a subtask.
        """
        self.subtasks[task.task_id] = task
        if task.parent_id is None:
            task.parent_id = self.task_id

    def _save(self):
        if self._save_file:
            with self._save_file.open("w") as f:
                print(self.model_dump_json(), file=f)

    def update_resources(self) -> ResourceMeasurement:
        """
        Update the resource measurements.  Returns the current measurement.

        This method is called by :meth:`update`, with an exclusive lock held.
        """
        assert self._initial_meter is not None
        now = ResourceMeasurement.current()
        elapsed = now - self._initial_meter
        self.duration = elapsed.perf_time
        self.cpu_time = elapsed.cpu_time
        self.peak_memory = elapsed.max_rss
        self.peak_gpu_memory = elapsed.max_gpu
        return now

    def __enter__(self):
        self.start()
        return self

    def __exit__(
        self, exc_type: type[Exception] | None, exc_value: Exception | None, traceback: Any
    ):
        log = _log.bind(task_id=str(self.task_id), label=self.label)
        if exc_type is None:
            status = TaskStatus.FINISHED
        elif issubclass(exc_type, KeyboardInterrupt):
            status = TaskStatus.CANCELLED
            log.error("task cancelled")
        else:
            status = TaskStatus.FAILED
            log.error("task failed: %s", exc_value)

        self.finish(status)

    def __reduce__(self):
        return self.__class__.model_validate, (self.model_dump(),)

    def __str__(self):
        return f"<Task {self.task_id}: {self.label}>"
