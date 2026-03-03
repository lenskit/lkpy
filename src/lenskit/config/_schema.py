# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from annotated_types import Gt, Le
from pydantic import BaseModel, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class PowerQueries(TypedDict, total=False):
    """
    Queries for requesting power consumption data from Prometheus.

    Each entry is a Python format string (for :meth:`str.format`), that will be
    used to format a dictionary ``{"elapsed": time_ms}``.  The result should be
    a valid Prometheus query that returns the power consumption in Joules (watt
    seconds) over the last ``time_ms`` milliseconds.
    """

    system: str
    "Total (chassis or system) power consumption."
    cpu: str
    "CPU power consumption."
    gpu: str
    "GPU power consumption."


class RandomSettings(BaseModel):
    """
    Random number generator configuration.
    """

    seed: int | None = None
    """
    The root RNG seed.
    """


class MachineSettings(BaseModel, extra="allow"):
    """
    Definition for a single machine.

    A “machine” is a computer (or cluster) that is in use for running LensKit
    experiments.  Many users won't use this, but if you want to use the power
    consumption monitoring, you will need to define how to measure power for the
    different machines in use.
    """

    description: str | None = None
    "Short description for this machine."
    power_queries: PowerQueries = {}
    "Prometheus queries to collect power metrics for this machine."


class PrometheusSettings(BaseModel):
    """
    Prometheus configuration settings.

    LensKit's task logging supports querying Prometheus for task-related metrics
    such as power consumption.
    """

    url: str | None = None


class TuneSettings(BaseModel):
    """
    LensKit hyperparameter tuning settings.
    """

    jobs: int | None = None
    """
    Number of allowed hyperparameter tuning jobs.
    """

    gpu_mult: Annotated[float, Gt(0), Le(1.0)] = 1.0
    """
    Multiplier for tuning job GPU requirements.  This is to coarsely adapt GPU
    requirements from configuration files to the local machine.  If a tuning
    specificataion requires 1 GPU, but your machine has enough capacity to run
    two jobs in parallel on a single GPU, you can set this to 0.5 to modify the
    tuning jobs to require 0.5 GPUs each.
    """

    max_points: int | None = None
    """
    Maximum number of search points for hyperparameter tuning.
    """


class ParallelSettings(BaseSettings):
    """
    Configuration for LensKit's parallel processing.  These settings are in the
    ``[parallel]`` table in ``lenskit.toml``.

    .. seealso::
        :ref:`parallel-config`
    """

    model_config = SettingsConfigDict(env_prefix="LK_")

    num_cpus: PositiveInt | None = None
    """
    The number of CPUs LensKit should consider using.  This is auto-detected
    from the system environment, and should only be configured manually if you
    want to override LensKit's CPU detection for some reason.  Note that the
    auto-detected values *do* account for operating system scheduling affinities
    and CPU limits.

    This value is not used directly as a limit, but is used to derive the
    default values for the other concurrency controls (threads, etc.).
    """

    use_ray: bool = False
    """
    Use Ray to parallelize batch operations, hyperparameter tuning, etc.
    """

    num_batch_jobs: int | None = None
    """
    Number of batch inference jobs to run in parallel.  Can be overridden with
    the :envvar:`LK_NUM_BATCH_JOBS` environment variable.
    """

    num_procs: int | None = None
    """
    Number of processes to use.
    """

    num_threads: int | None = None
    """
    Number of threads to use.  Can be overridden with the
    :envvar:`LK_NUM_THREADS` environment variable.  Specify -1 to use all
    available threads.
    """

    num_backend_threads: int | None = None
    """
    Number of threads for compute backends to use.  Can be overridden with the
    :envvar:`LK_NUM_BACKEND_THREADS` environment variable.  Specify -1 to leave
    threading limits unmodified.
    """

    @property
    def total_threads(self):
        nt = self.resolved_num_threads
        nbt = self.resolved_num_backend_threads
        if nbt is None:
            return nt
        else:
            return nt * nbt

    @property
    def usable_cpus(self) -> int:
        """
        Get the number of available CPUs, from :attr:`num_cpus` or the system.
        """
        from lenskit.parallel import effective_cpu_count

        if self.num_cpus is None:
            return effective_cpu_count()
        else:
            return self.num_cpus

    @property
    def resolved_num_threads(self) -> int:
        """
        Get the number of compute threads to use, resolving defaults.
        """
        ncpu = self.usable_cpus
        if self.num_threads is None:
            return min(ncpu, 8)
        elif self.num_threads <= 0:
            return ncpu
        else:
            return self.num_threads

    @property
    def resolved_num_backend_threads(self) -> int | None:
        """
        Get the number of backend threads to use, resolving defaults.

        Returns:
            The number of backend threads, or ``None`` to leave backends
            unconfigured.
        """
        if self.num_backend_threads is None:
            return max(min(self.usable_cpus // self.resolved_num_threads, 4), 1)
        elif self.num_backend_threads <= 0:
            return None
        else:
            return self.num_backend_threads

    @property
    def resolved_num_batch_jobs(self) -> int:
        from lenskit.parallel import is_free_threaded

        if self.num_batch_jobs is None:
            if is_free_threaded():
                return self.resolved_num_threads
            else:
                return 1
        elif self.num_batch_jobs <= 0:
            return self.usable_cpus
        else:
            return self.num_batch_jobs

    def env_vars(self) -> dict[str, str]:
        """
        Get the parallel configuration as a set of environment variables.  The
        set also includes ``OMP_NUM_THREADS`` and related variables for BLAS and
        MKL to configure OMP early.
        """
        evs = {
            "LK_NUM_BATCH_JOBS": str(self.resolved_num_batch_jobs),
            "LK_NUM_THREADS": str(self.resolved_num_threads),
            "LK_NUM_BACKEND_THREADS": str(self.resolved_num_backend_threads or -1),
        }
        if "OMP_NUM_THREADS" not in os.environ:
            evs["OMP_NUM_THREADS"] = evs["LK_NUM_BACKEND_THREADS"]
        if "OPENBLAS_NUM_THREADS" not in os.environ:
            evs["OPENBLAS_NUM_THREADS"] = evs["LK_NUM_BACKEND_THREADS"]
        if "MKL_NUM_THREADS" not in os.environ:
            evs["MKL_NUM_THREADS"] = evs["LK_NUM_BACKEND_THREADS"]
        return evs


class LenskitSettings(BaseSettings, extra="allow"):
    """
    Definition of LensKit settings.

    LensKit supports loading various settings from configuration files
    and the environment as a consistent way to control LensKit's various
    control surfaces.

    Stability:
        Experimental
    """

    model_config = SettingsConfigDict(
        nested_model_default_partial_update=True, env_prefix="LK_", env_nested_delimiter="__"
    )

    random: RandomSettings = RandomSettings()
    """
    Random number generator configuration.
    """

    parallel: ParallelSettings = ParallelSettings()

    machine: str | None = None
    """
    The name of the machine running experiments.

    This is usually set in ``lenskit.local.toml`` or the ``LK_MACHINE`` environment
    variable.
    """
    prometheus: PrometheusSettings = PrometheusSettings()
    """
    Prometheus settings for task metric collection.
    """
    machines: dict[str, MachineSettings] = {}
    """
    Description of different machines used in the experiment(s), to support
    things like collecting power metrics.
    """

    tuning: TuneSettings = TuneSettings()
    """
    LensKit tuning settings.
    """

    @property
    def current_machine(self) -> MachineSettings | None:
        if self.machine:
            if ms := self.machines.get(self.machine, None):
                return ms

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
