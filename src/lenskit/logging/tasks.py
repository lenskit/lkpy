# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Abstraction for recording tasks.
"""

# pyright: strict
from __future__ import annotations

import socket
from enum import Enum
from os import PathLike
from pathlib import Path
from threading import Lock
from typing import Annotated, Any
from uuid import UUID, uuid4

import requests
from pydantic import AliasChoices, BaseModel, BeforeValidator, Field, SerializeAsAny
from typing_extensions import Literal

from ._proxy import get_logger
from .formats import friendly_duration
from .resource import ResourceMeasurement, reset_linux_hwm

_log = get_logger(__name__)
_active_tasks: list[Task] = []


def _dict_extract_values(data: object) -> Any:
    if isinstance(data, dict):
        return list(data.values())  # type: ignore
    else:
        return data


def _lk_machine():
    from lenskit.config import lenskit_config

    return lenskit_config().machine


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

    Stability:
        Caller

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
    parent_id: UUID | None = None
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
    hostname: str = Field(default_factory=socket.gethostname)
    """
    The hostname on which the task was run.
    """
    machine: str | None = Field(default_factory=_lk_machine)
    """
    The machine on which the task was run.

    Machines are project-meaningful labels for compute machines or clusters,
    primarily for understanding resource consumption.  See :ref:`settings`.
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

    system_power: float | None = Field(
        default=None, validation_alias=AliasChoices("system_power", "chassis_power")
    )
    """
    Estimated total system power consumption (in Joules).
    """
    cpu_power: float | None = None
    """
    Estimated CPU power consumption (in Joules).
    """
    gpu_power: float | None = None
    """
    Estimated GPU power consumption (in Joules).
    """

    subtasks: Annotated[list[SerializeAsAny[Task]], BeforeValidator(_dict_extract_values)] = Field(  # type: ignore
        default_factory=list
    )
    """
    This task's subtasks.
    """

    _reset_hwm: bool = False
    _save_file: Path | None = None
    _lock: Lock = None  # type: ignore
    _initial_meter: ResourceMeasurement | None = None
    _final_meter: ResourceMeasurement | None = None
    _refresh_id: UUID | None = None
    _subtask_index: dict[UUID, Task] | None = None

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

    def total_cpu(self) -> float | None:
        "Compute the total CPU time (including subprocesses)."
        time = self.cpu_time
        if time is None:
            return None

        for task in self.subtasks:
            if task.subprocess:
                child_time = task.total_cpu()
                if child_time is not None:
                    time += child_time

        return time

    @property
    def friendly_duration(self) -> str | None:
        if self.duration is None:
            return None
        else:
            return friendly_duration(self.duration)

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
        _log.debug("adding subtask", task=task.model_dump())
        with self._lock:
            if self._subtask_index is None:
                self._subtask_index = {t.task_id: t for t in self.subtasks}
            self._subtask_index[task.task_id] = task

            if task.parent_id is None:
                task.parent_id = self.task_id

            self.subtasks = list(self._subtask_index.values())

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

        self.system_power = measure_power("system", elapsed.perf_time)
        self.cpu_power = measure_power("cpu", elapsed.perf_time)
        self.gpu_power = measure_power("gpu", elapsed.perf_time)

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
            log.debug("task cancelled")
        else:
            status = TaskStatus.FAILED
            log.debug("task failed: %s", exc_value)

        self.finish(status)

    def __reduce__(self):
        return self.__class__.model_validate, (self.model_dump(),)

    def __str__(self):
        return f"<Task {self.task_id}: {self.label}>"


def measure_power(scope: Literal["system", "cpu", "gpu"], duration: float):
    from lenskit.config import lenskit_config

    time_ms = int(duration * 1000)

    config = lenskit_config()
    prom = config.prometheus.url
    if not prom:
        return None

    machine = config.current_machine
    if machine is None:
        return None

    url = prom + "/api/v1/query"
    if query := machine.power_queries.get(scope):
        query = query.format(elapsed=time_ms, machine=config.machine)
        return _get_prometheus_metric(url, query, time_ms)


def _get_prometheus_metric(url: str, query: str, time_ms: int) -> float | None:
    log = _log.bind(url=url, query=query)
    try:
        res = requests.get(url, {"query": query}).json()
    except Exception as e:
        log.warning("Prometheus query error", exc_info=e)
        return None

    log.debug("received response", response=res)
    if res["status"] == "error":
        log.error("Prometheus query error: %s", res["error"], type=res["errorType"])
        return None

    results = res["data"]["result"]
    if len(results) == 0:
        log.debug("Prometheus query returned no results")
        return None
    elif len(results) > 1:
        log.error("Prometheus query returned %d results, expected 1", len(results))

    _time, val = results[0]["value"]
    return float(val)
