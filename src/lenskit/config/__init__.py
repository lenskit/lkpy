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
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from os import PathLike
from pathlib import Path

from pydantic_settings import TomlConfigSettingsSource
from typing_extensions import Any, TypeVar, overload

from lenskit.diagnostics import ConfigWarning
from lenskit.logging import get_logger
from lenskit.random import init_global_rng
from lenskit.schemas.settings import (
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
_context_settings: ContextVar[LenskitSettings | None] = ContextVar("lenskit.settings", default=None)


def lenskit_config() -> LenskitSettings:
    """
    Get the LensKit configuration.

    If no configuration has been specified, returns a default settings object.
    """
    global _settings

    settings = _context_settings.get()
    if settings is None:
        settings = _settings

    if settings is None:
        settings = LenskitSettings()
        settings.finish_setup()

    return settings


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
    defaults and environment variables.

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


@overload
def reconfigure(cfg_dir: Path | None = None) -> AbstractContextManager[LenskitSettings]: ...
@overload
def reconfigure(
    cfg_dir: Path | None = None, *, settings_cls: type[SettingsClass]
) -> AbstractContextManager[LenskitSettings]: ...
@overload
def reconfigure(settings: LenskitSettings, /) -> AbstractContextManager[LenskitSettings]: ...
@contextmanager
def reconfigure(
    cfg_dir: Path | LenskitSettings | None = None,
    *,
    settings_cls=LenskitSettings,
):
    """
    Temporarily reconfigure LensKit, overriding the existing configuration.

    .. note::

        This overwrites the active :class:`LenskitSettings`, but does *not*
        reconfigure parallelism or other globally-configured state.  It is
        mostly useful for tests to inject a controlled LensKit settings to
        test how other code responds to those settings.

    .. seealso:: :func:`configure`

    Returns:
        A context manager that will restore the previous settings state when
        exited.

    Stability:
        Internal
    """
    settings = configure(cfg_dir, settings_cls=settings_cls, _set_global=False)  # ty:ignore[no-matching-overload]
    token = _context_settings.set(settings)
    try:
        yield settings
    finally:
        _context_settings.reset(token)


def _load_settings[C](cfg_dir: Path | None, settings_cls: type[C]) -> C:
    # define a subclass so we can specify the configuration location
    toml_files = ["lenskit.toml", "lenskit.local.toml"]
    if cfg_dir is not None:
        toml_files = [cfg_dir / f for f in toml_files]

    class LenskitFileSettings(settings_cls):  # ty:ignore[shadowed-type-variable, unsupported-base]
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

    obj = LenskitFileSettings()
    obj.finish_setup()
    return obj  # type: ignore


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
