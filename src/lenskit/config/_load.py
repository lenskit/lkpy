# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import json
import tomllib
from os import PathLike
from pathlib import Path
from typing import overload

from pydantic import BaseModel, JsonValue

from lenskit.logging import get_logger

_log = get_logger(__name__)


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
def load_config_data[M: BaseModel](path: Path | PathLike[str], model: type[M]) -> M: ...
def load_config_data[M: BaseModel](path: Path | PathLike[str], model: type[M] | None = None):
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
