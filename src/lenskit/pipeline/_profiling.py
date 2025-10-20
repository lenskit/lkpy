# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from csv import DictWriter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING, Protocol, TextIO
from uuid import UUID, uuid4

from xopen import xopen

from lenskit.logging import get_logger
from lenskit.logging.multiprocess import Monitor, RecordSink, WorkerContext, get_monitor

if TYPE_CHECKING:
    from ._impl import Pipeline

_log = get_logger(__name__)


class ProfileSink(RecordSink[dict[str, float]], Protocol):
    def run_profiler(self) -> RunProfiler:
        return RunProfiler(self)


class PipelineProfiler:
    """
    Collect pipeline run statistics for profiling pipeline executions.
    """

    writer: DictWriter[str]
    output: TextIO
    sink_id: UUID
    _monitor: Monitor | None = None

    def __init__(self, pipeline: Pipeline, file: Path):
        self.sink_id = uuid4()
        self.output = xopen(file, "wt")
        stages = pipeline.component_names()
        _log.debug("starting pipeline profiler %s", self.sink_id)
        self.writer = DictWriter(self.output, stages)
        self.writer.writeheader()

    def set_stages(self, stages: list[str]):
        self.stages = stages
        if self.output:
            self.writer = DictWriter(self.output, stages)
            self.writer.writeheader()

    def multiprocess(self) -> ProfileSink:
        _log.debug("registering profiler %s with monitor", self.sink_id)
        self._monitor = get_monitor()
        self._monitor.add_record_sink(self)
        return WorkerProfileSink(self.sink_id)

    def close(self):
        if self._monitor is not None:
            _log.debug("waiting for monitor to quiesce")
            self._monitor.await_quiesce()

        self.output.close()
        if self._monitor is not None:
            self._monitor.remove_record_sink(self.sink_id)
            self._monitor = None

    def record(self, record: dict[str, float]):
        self.writer.writerow(record)

    def run_profiler(self):
        return RunProfiler(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass
class WorkerProfileSink:
    sink_id: UUID

    def record(self, record: dict[str, float]):
        ctx = WorkerContext.active()
        if ctx is not None:
            ctx.send_record(self.sink_id, record)

    def run_profiler(self):
        return RunProfiler(self)


class RunProfiler:
    """
    Internal utility class for profiling a single pipeline run.
    """

    profiler: ProfileSink
    _record: dict[str, float]
    _starts: dict[str, float]

    def __init__(self, profiler: ProfileSink):
        self.profiler = profiler
        self._record = {}
        self._starts = {}

    def start_stage(self, stage: str):
        self._starts[stage] = perf_counter_ns()

    def finish_stage(self, stage: str):
        now = perf_counter_ns()
        start = self._starts[stage]

        self._record[stage] = now - start

    def finish_run(self):
        self.profiler.record(self._record)
