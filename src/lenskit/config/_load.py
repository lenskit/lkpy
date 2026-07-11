# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from os import PathLike
from pathlib import Path

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
