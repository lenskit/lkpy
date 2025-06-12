# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit general configuration
"""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource


class PowerQueries(TypedDict, total=False):
    """
    Queries for requesting power consumption data from Prometheus.

    Each entry is a Python format string (for :meth:`str.format`), that will be
    used to format a dictionary ``{"elapsed": time_ms}``.  The result should be
    a valid Prometheus query that returns the power consumption in Joules (watt
    seconds) over the last ``time_ms`` milliseconds.
    """

    total: str
    "Total (chassis or system) power consumption."
    cpu: str
    "CPU power consumption."
    gpu: str
    "GPU power consumption."


class RandomConfig(BaseModel):
    """
    Random number generator configuration.
    """

    seed: int | None = None
    """
    The root RNG seed.
    """


class Machine(BaseModel, extra="allow"):
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
        nested_model_default_partial_update=True,
        env_prefix="LK_",
        toml_file="lenskit.toml",
    )

    random: RandomConfig
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
    machines: dict[str, Machine] = {}
    """
    Description of different machines used in the experiment(s), to support
    things like collecting power metrics.
    """

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            TomlConfigSettingsSource(settings_cls),
        )
