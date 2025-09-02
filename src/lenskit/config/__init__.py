# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit general configuration
"""

from __future__ import annotations

import json
import tomllib
import warnings
from os import PathLike
from pathlib import Path

from pydantic import BaseModel, JsonValue
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource
from typing_extensions import Any, TypedDict, TypeVar, overload

from lenskit.diagnostics import ConfigWarning
from lenskit.logging import get_logger
from lenskit.random import init_global_rng

__all__ = [
    "lenskit_config",
    "configure",
    "LenskitSettings",
    "RandomSettings",
    "MachineSettings",
    "PowerQueries",
    "PrometheusSettings",
]

M = TypeVar("M", bound=BaseModel)
"Model class for general configuration loading."
SettingsClass = TypeVar("SettingsClass", bound="LenskitSettings", default="LenskitSettings")
_log = get_logger(__name__)
_settings: LenskitSettings | None = None


def lenskit_config() -> LenskitSettings:
    """
    Get the LensKit configuration.

    If no configuration has been specified, returns a default settings object.
    """
    global _settings

    if _settings is None:
        return LenskitSettings()
    else:
        return _settings


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


@overload
def configure(cfg_dir: Path | None = None) -> LenskitSettings: ...
@overload
def configure(
    cfg_dir: Path | None = None, *, settings_cls: type[SettingsClass]
) -> SettingsClass: ...
def configure(
    cfg_dir: Path | None = None, *, settings_cls=LenskitSettings, _set_global: bool = True
) -> Any:
    """
    Initialize LensKit configuration.

    LensKit does **not** automatically read configuration files — if this
    function is never called, then configuration will entirely be done through
    defaults and environment varibles.

    This function will automatically configure the global RNG, if a seed is
    specified.

    Args:
        cfg_dir:
            The directory in which to look for configuration files..  If not
            provided, uses the current directory.
        settings_cls:
            The base LensKit settings class.  Rarely used, only needed if a
            project wants to to extend LensKit settings with their own settings.

    Returns:
        The configured LensKit settings.
    """
    global _settings

    if _settings is not None and _set_global:
        warnings.warn("LensKit already configured, overwriting configuration", ConfigWarning)

    # define a subclass so we can specify the configuration location
    toml_files = ["lenskit.toml", "lenskit.local.toml"]
    if cfg_dir is not None:
        toml_files = [cfg_dir / f for f in toml_files]

    class LenskitFileSettings(settings_cls):
        @classmethod
        def settings_customise_sources(
            cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
        ):
            return (
                init_settings,
                env_settings,
                dotenv_settings,
                file_secret_settings,
                TomlConfigSettingsSource(
                    settings_cls,
                    cfg_dir / "lenskit.local.toml" if cfg_dir is not None else "lenskit.local.toml",
                ),
                TomlConfigSettingsSource(
                    settings_cls,
                    cfg_dir / "lenskit.toml" if cfg_dir is not None else "lenskit.toml",
                ),
            )

    settings = LenskitFileSettings()
    if _set_global:
        _settings = settings
        if settings.random.seed is not None:
            init_global_rng(settings.random.seed)

    return settings


def locate_configuration_root(
    *,
    cwd: Path | str | PathLike[str] | None = None,
    abort_at_pyproject: bool = True,
    abort_at_gitroot: bool = True,
) -> Path | None:
    """
    Search for a configuration root containing a ``lenskit.toml`` file.

    This searches for a ``lenskit.toml`` file, beginning in the current working
    directory (or the alternate ``cwd`` if provided), and searching upward until
    one is found.  Search stops if a ``pyproject.toml`` file or ``.git``
    directory is found without encountering ``lenskit.toml``.
    """

    if cwd is None:
        cwd = Path()
    elif not isinstance(cwd, Path):
        cwd = Path(cwd)
    cwd = cwd.resolve()

    log = _log.bind(cwd=str(cwd))
    log.debug("searching for lenskit.toml")
    while cwd is not None:
        log.debug("checking if lenskit.toml exists", dir=str(cwd))
        if (cwd / "lenskit.toml").exists():
            return cwd

        if abort_at_pyproject and (cwd / "pyproject.toml").exists():
            break

        if abort_at_gitroot and (cwd / ".git").exists():
            break

        if cwd.parent == cwd:
            break
        else:
            cwd = cwd.parent


@overload
def load_config_data(path: Path | PathLike[str], model: None = None) -> JsonValue: ...
@overload
def load_config_data(path: Path | PathLike[str], model: type[M]) -> M: ...
def load_config_data(path: Path | PathLike[str], model: type[M] | None = None):
    """
    General-purpose function to automatically load configuration data and
    optionally validate with a model.

    Args:
        path:
            The path to the configuration file.
        model:
            The Pydantic model class to validate.
    """
    path = Path(path)
    text = path.read_text()

    match path.suffix:
        case ".json" if model is not None:
            return model.model_validate_json(text)
        case ".json":
            data = json.loads(text)
        case ".toml":
            data = tomllib.loads(text)
        case ".yaml" | ".yml":
            import yaml

            data = yaml.load(text, yaml.SafeLoader)

        case _:
            raise ValueError(f"unsupported configuration type for {path}")

    if model is None:
        return data
    else:
        return model.model_validate(data)
