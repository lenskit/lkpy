# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Annotated, TypedDict

from annotated_types import Gt, Le
from pydantic import BaseModel
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
