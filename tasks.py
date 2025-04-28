# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import os
import os.path
import re
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from shutil import copyfileobj, rmtree
from urllib.request import urlopen

import tomlkit
from invoke.context import Context
from invoke.main import program
from invoke.tasks import task

CACHEDIR_TAG = "Signature: 8a477f597d28d172789f06886806bc55"
BIBTEX_PATH = "http://127.0.0.1:23119/better-bibtex/export/collection?/4/9JMHQD9K.bibtex"

if sys.stdout.isatty():
    os.environ["CLICOLOR_FORCE"] = "1"
    os.environ["FORCE_COLOR"] = "1"

_log = logging.getLogger("lenskit.invoke")
root = Path(__file__).parent

try:
    from lenskit._version import lk_git_version
except ImportError:
    sys.path.insert(0, os.fspath(root.absolute() / "src"))
    from lenskit._version import lk_git_version


def _update_version(c: Context, write: bool = False):
    ver = lk_git_version()

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
    ver = lk_git_version()
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

    version = lk_git_version()
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
def update_headers(
    c: Context,
    year: int | None = None,
    check_only: bool = False,
    error_on_change: bool = False,
):
    "Update or check license headers."
    from unbeheader.headers import SUPPORTED_FILE_TYPES, update_header
    from unbeheader.typing import CommentSkeleton, SupportedFileType

    if year is None:
        today = date.today()
        year = today.year

    SUPPORTED_FILE_TYPES["rs"] = SupportedFileType(
        re.compile(r"((^//|[\r\n]//).*)*"),
        CommentSkeleton("//", "//"),
    )

    if program.core.remainder:
        files = [Path(p) for p in re.split(r"\s", program.core.remainder)]
    else:
        src = Path("src")
        files = list(src.glob("**/*.py"))
        files += src.glob("**/*.rs")

    n = 0
    for file in files:
        if update_header(file, year, check=check_only):
            n += 1

    print("updated", n, "files")
    if error_on_change and n > 0:
        sys.exit(5)


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
