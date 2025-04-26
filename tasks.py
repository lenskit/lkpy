# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import os
import os.path
import re
import sys
import tomllib
from contextlib import contextmanager
from pathlib import Path
from shutil import copyfileobj, rmtree
from urllib.request import urlopen

import tomlkit
from invoke.context import Context
from invoke.main import program
from invoke.tasks import task
from packaging.version import Version
from packaging.version import parse as parse_version

CACHEDIR_TAG = "Signature: 8a477f597d28d172789f06886806bc55"
BIBTEX_PATH = "http://127.0.0.1:23119/better-bibtex/export/collection?/4/9JMHQD9K.bibtex"

if sys.stdout.isatty():
    os.environ["CLICOLOR_FORCE"] = "1"
    os.environ["FORCE_COLOR"] = "1"

_log = logging.getLogger("lenskit.invoke")
root = Path(__file__).parent


def _get_version(c: Context) -> Version:
    gd = c.run('git describe --tags --match "v*"', hide="out")
    assert gd is not None
    assert gd.stdout is not None
    ver = gd.stdout.strip()
    m = re.match(r"^v(\d+\.\d+\.\d+[.a-z0-9]*)(?:-(\d+)-(g[0-9a-fA-F]+))?$", ver)
    if not m:
        raise ValueError(f"unparseable version: {ver}")

    if m.group(2):
        pvs = f"{m.group(1)}.dev{m.group(2)}+{m.group(3)}"
    else:
        pvs = ver[1:]

    _log.debug("parsing %s", pvs)
    version = parse_version(pvs)

    with open("Cargo.toml", "rb") as tf:
        cargo = tomllib.load(tf)
    cv = cargo["package"]["version"]
    if m := re.match(r"(.*)-([a-z]+)(?:\.(\d+))", cv):
        cv_pr = m.group(2)
        if cv_pr != "rc":
            cv_pr = cv_pr[:1]
        cv_py = m.group(1) + m.group(2) + m.group(3)
    else:
        cv_py = cv

    cv_ver = parse_version(cv_py)

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
            _log.warning("version mismatch: cargo {} != git {}", version, cv_ver)

    return version


def _update_version(c: Context, write: bool = False):
    ver = _get_version(c)

    with open("pyproject.toml", "rt") as tf:
        meta = tomlkit.load(tf)

    proj = meta["project"]
    proj["dynamic"].remove("version")
    proj["version"] = str(ver)
    if write:
        with open("pyproject.toml", "wt") as tf:
            tomlkit.dump(meta, tf)
    else:
        tomlkit.dump(meta, sys.stdout)

    return ver


@contextmanager
def _updated_pyproject_toml(c: Context):
    """
    Context manager that updates the pyproject version, then restores it.
    """
    file = Path("pyproject.toml")
    old = file.read_bytes()
    try:
        ver = _update_version(c, write=True)
        yield ver
    finally:
        file.write_bytes(old)


def _make_cache_dir(path: str | Path):
    "Create a directory and a CACHEDIR.TAG file."
    path = Path(path)
    path.mkdir(exist_ok=True)
    with open(path / "CACHEDIR.TAG", "wt") as ctf:
        print(CACHEDIR_TAG, file=ctf)
        print("# Cache directory marker for LensKit build", file=ctf)
        print(
            "# For information about cache directory tags see https://bford.info/cachedir/",
            file=ctf,
        )


@task
def version(c: Context):
    ver = _get_version(c)
    print(ver)


@task
def update_pypi_version(c: Context, write=False):
    _update_version(c, write=write)


@task
def setup_dirs(c: Context):
    "Initialize output directories."
    _make_cache_dir("dist")
    _make_cache_dir("build")
    _make_cache_dir("output")


@task(setup_dirs)
def build_sdist(c: Context):
    "Build source distribution."
    with _updated_pyproject_toml(c) as ver:
        print("packaging LensKit version", ver)
        c.run("uv build --sdist")

        if gh_file := os.environ.get("GITHUB_OUTPUT", None):
            with open(gh_file, "at") as ghf:
                print(f"version={ver}", file=ghf)


@task(setup_dirs)
def build_dist(c: Context):
    "Build packages for the current platform."
    c.run("uv build")


@task(setup_dirs)
def build_accel(c: Context, release: bool = False):
    "Build the accelerator in-place."
    cmd = "maturin develop"
    if release:
        cmd += " --release"
    c.run(cmd, echo=True)


@task(build_sdist)
def build_conda(c: Context):
    "Build Conda packages."

    version = _get_version(c)
    print("building Conda packages for LensKit version {}", version)
    cmd = "rattler-build build --recipe conda --output-dir dist/conda"
    if "CI" in os.environ:
        cmd += " --noarch-build-platform linux-64"
    c.run(cmd, echo=True, env={"LK_PACKAGE_VERSION": str(version)})


@task(build_accel, positional=["file"])
def test(
    c: Context, coverage: bool = False, skip_marked: str | None = None, file: str | None = None
):
    "Run tests."
    cmd = "pytest"
    if coverage:
        cmd += " --cov=src/lenskit --cov-report=term --cov-report=xml"
    if skip_marked:
        cmd += f" -m 'not {skip_marked}'"
    if file:
        cmd += f" '{file}'"

    if program.core.remainder:
        cmd += " " + program.core.remainder
    else:
        cmd += " tests"

    c.run(cmd, echo=True)


@task(setup_dirs)
def docs(c: Context):
    "Build documentation."
    c.run("sphinx-build docs build/doc")


@task(setup_dirs)
def preview_docs(c: Context):
    "Auto-build and preview documentation."
    c.run("sphinx-autobuild --watch src docs build/doc")


@task
def update_bibtex(c: Context):
    "Update BibTeX file."
    print("fetching BibTeX")
    with urlopen(BIBTEX_PATH) as src, open("docs/lenskit.bib", "wb") as dst:
        copyfileobj(src, dst)


@task
def update_headers(c: Context):
    c.run("unbehead", echo=True)


@task
def clean(c: Context):
    print(c.config)
    for od in ["build", "dist", "output", "target"]:
        odp = root / od
        if odp.exists():
            print(f"🚮 removing {od}/   ")
            if not c.config.run.dry:
                rmtree(odp, ignore_errors=True)

    for glob in ["*.lprof", "*.profraw", "*.prof", "*.log"]:
        print(f"🚮 removing {glob}")
        for file in root.glob(glob):
            _log.info("removing %s", file)
            if not c.config.run.dry:
                file.unlink()

    print("cleaning generated doc files")
    c.run("git clean -xf docs")
