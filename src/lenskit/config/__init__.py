# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit general configuration
"""

from __future__ import annotations

import warnings
from pathlib import Path

from pydantic_settings import TomlConfigSettingsSource
from typing_extensions import Any, TypeVar, overload

from lenskit.diagnostics import ConfigWarning
from lenskit.logging import get_logger
from lenskit.random import init_global_rng

from ._load import load_config_data, locate_configuration_root
from ._schema import (
    LenskitSettings,
    MachineSettings,
    ParallelSettings,
    PowerQueries,
    PrometheusSettings,
    RandomSettings,
    TuneSettings,
)

__all__ = [
    "lenskit_config",
    "load_config_data",
    "locate_configuration_root",
    "configure",
    "LenskitSettings",
    "RandomSettings",
    "MachineSettings",
    "ParallelSettings",
    "PowerQueries",
    "PrometheusSettings",
    "TuneSettings",
]

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


@overload
def configure(cfg_dir: Path | None = None) -> LenskitSettings: ...
@overload
def configure(
    cfg_dir: Path | None = None, *, settings_cls: type[SettingsClass]
) -> SettingsClass: ...
@overload
def configure(settings: LenskitSettings, /) -> LenskitSettings: ...
def configure(
    cfg_dir: Path | LenskitSettings | None = None,
    *,
    settings_cls=LenskitSettings,
    _set_global: bool = True,
) -> Any:
    """
    Initialize LensKit configuration.

    LensKit does **not** automatically read configuration files — if this
    function is never called, then configuration will entirely be done through
    defaults and environment varibles.

    This function will automatically configure the global RNG, if a seed is
    specified.  It does **not** configure logging.

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

    if isinstance(cfg_dir, LenskitSettings):
        settings = cfg_dir
    else:
        settings = _load_settings(cfg_dir, settings_cls)

    if _set_global:
        _settings = settings
        if settings.random.seed is not None:
            init_global_rng(settings.random.seed)
        from lenskit.parallel import init_threading

        init_threading(settings.parallel)

    return settings


def _load_settings[SettingsClass](
    cfg_dir: Path | None, settings_cls: type[SettingsClass]
) -> SettingsClass:
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

    return LenskitFileSettings()  # type: ignore
