# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import re
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from subprocess import CalledProcessError, check_output
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packaging.version import Version

_log = logging.getLogger(__name__)
_lk_mod_dir = Path(__file__).parent
_lk_root = _lk_mod_dir.parent.parent


def is_git_install() -> bool:
    if _lk_mod_dir.parent.name != "src":
        return False

    git = _lk_mod_dir.parent.parent / ".git"
    return git.exists()


def lk_git_version() -> Version:
    from packaging.version import Version
    from packaging.version import parse as parse_version

    gd = check_output(["git", "describe", "--tags", "--match", "v*"], cwd=_lk_mod_dir.parent.parent)
    ver = gd.decode().strip()
    _log.debug("parsing Git info %s", ver)
    m = re.match(r"^v(\d+\.\d+\.\d+[.a-z0-9]*)(?:-(\d+)-(g[0-9a-fA-F]+))?$", ver)
    if not m:
        raise ValueError(f"unparseable version: {ver}")

    if m.group(2):
        pvs = f"{m.group(1)}.dev{m.group(2)}+{m.group(3)}"
    else:
        pvs = ver[1:]

    _log.debug("parsing transformed Git %s", pvs)
    version = parse_version(pvs)
    _log.info("Git version: %s", version)

    with open(_lk_root / "Cargo.toml", "rb") as tf:
        cargo = tomllib.load(tf)
    cv = cargo["package"]["version"]
    _log.debug("parsing Cargo version %s", cv)
    if m := re.match(r"(.*)-([a-z]+)(?:\.(\d+))(\.post\d+)", cv):
        cv_pr = m.group(2)
        if cv_pr != "rc":
            cv_pr = cv_pr[:1]
        cv_py = m.group(1) + m.group(2) + m.group(3) + m.group(4)
    else:
        cv_py = cv

    _log.debug("parsing transformed Cargo version %s", cv_py)
    cv_ver = parse_version(cv_py)
    _log.info("Cargo version: %s", cv_ver)

    if version.is_devrelease:
        if cv_ver > version:
            _log.debug("cargo requested version is newer")
            base = cv_ver.public
        else:
            _log.warning("Cargo version %s older than Git %s", cv_ver, version)
            if version.is_prerelease:
                assert version.pre is not None
                base = version.base_version + version.pre[0] + str(version.pre[1] + 1)
            else:
                base = f"{version.major}.{version.minor}.{version.micro}"

        version = Version(f"{base}.dev{version.dev}+{version.local}")
    else:
        if version != cv_ver:
            _log.warning("version mismatch: cargo %s != git %s", cv_ver, version)

    return version


def lenskit_version() -> str:
    if is_git_install():
        try:
            return str(lk_git_version())
        except CalledProcessError:
            pass

    try:
        return version("lenskit")
    except PackageNotFoundError:  # pragma: nocover
        return "UNKNOWN"
