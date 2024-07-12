#!/usr/bin/env python
# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Generate and manage Conda environments.

Usage:
    lkdev.conda [-v] [-o FILE | -n NAME]
        [--micromamba] [--mkl] [--cuda]
        [-p VER] [-e EXTRA]... REQFILE...

Options:
    -v, --verbose   enable verbose logging
    -o FILE, --output=FILE
                    write output to FILE
    -n NAME, --name=NAME
                    create environment NAME
    -p VER, --python-version=VER
                    use Python version VER
    -e EXTRA, --extra=EXTRA
                    include extra EXTRA ('all' to include all extras)
    --micromamba    use micromamba
    --mkl           enable MKL BLAS
    --cuda          enable CUDA PyTorch

This tool is implemented as its own script, because it is used to
bootstrap LensKit and therefore cannot depend on LensKit being installed.
"""

# pyright: strict, reportPrivateUsage=false
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess as sp
import sys
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Generator, Iterable, Iterator, NamedTuple, Optional

import tomlkit
import tomlkit.items
import yaml
from docopt import docopt
from packaging.requirements import Requirement

_log = logging.getLogger("conda-tool")

CONDA_CHANNELS = ["conda-forge", "pytorch", "nodefaults"]
MKL_DEP = ["libblas=*=*mkl*"]
ALL_PYTHONS = ["3.10", "3.11"]
ALL_VARIANTS = {
    "ci": ["pyproject.toml", "test-requirements.txt"],
    "dev": ["pyproject.toml", "dev-requirements.txt"],
    "doc": ["pyproject.toml", "doc-requirements.txt"],
}


class ParsedReq(NamedTuple):
    source: Path
    name: str
    requirement: Optional[Requirement] = None
    conda: Optional[tuple[str, Optional[str]]] = None
    force_pip: bool = False

    def conda_spec(self) -> str | None:
        name = None
        ver = None
        if self.requirement:
            name = self.requirement.name
        if self.conda:
            name, ver = self.conda

        if self.requirement:
            pip_spec = self.requirement.specifier
            if ver is None:
                ver = ",".join(str(s) for s in pip_spec._specs)

        if ver:
            return f"{name} {ver}"
        else:
            return name

    def __str__(self):
        msg = str(self.requirement)
        if self.conda:
            name, ver = self.conda
            msg += " (conda: {}{})".format(name, ver if ver else "")
        if self.force_pip:
            msg += " (pip)"
        return msg


def main():
    global options
    options = docopt(__doc__)
    level = logging.DEBUG if options["--verbose"] else logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)

    specs = [
        r for r in load_reqfiles(options["REQFILE"]) if not re.match(r"lenskit($|-\w+)$", r.name)
    ]
    env = make_env_object(specs, options["--python-version"])
    if options["--name"]:
        tmpfd, tmpf = mkstemp(suffix="-env.yml", prefix="lkpy-")
        try:
            _log.debug("saving environment to %s", tmpf)
            with os.fdopen(tmpfd, "wt") as tf:
                yaml.safe_dump(env, tf)
            _log.debug("creating environment")
            conda = find_conda_executable()
            sp.check_call([conda, "env", "create", "-n", options["--name"], "-f", tmpf])
        finally:
            os.unlink(tmpf)
    elif options["--output"]:
        _log.info("writing to file %s", options["--output"])
        with open(options["--output"], "wt") as f:
            yaml.safe_dump(env, f)
    else:
        yaml.safe_dump(env, sys.stdout)


def load_reqfiles(files: Iterable[Path | str]) -> Iterator[ParsedReq]:
    for file in files:
        yield from load_requirements(Path(file))


def load_requirements(file: Path) -> list[ParsedReq]:
    if file.name == "pyproject.toml":
        return load_req_toml(file)
    elif re.match(r".*\.ya?ml$", file.name):
        return load_conda_yml(file)
    else:
        return load_req_txt(file)


def load_req_toml(file: Path) -> list[ParsedReq]:
    _log.info("loading project file %s", file)
    text = file.read_text()
    parsed = tomlkit.parse(text)
    proj = parsed.item("project")
    assert isinstance(proj, tomlkit.items.Table)
    deps = proj["dependencies"]
    assert isinstance(deps, tomlkit.items.Array)

    lines = list(array_lines(deps))

    extras = options["--extra"]
    if "optional-dependencies" in proj:
        edeps = proj["optional-dependencies"]
        assert isinstance(edeps, tomlkit.items.Table)
        for e in edeps.keys():  # type: ignore
            assert isinstance(e, str)
            _log.debug("checking extra %s", e)
            if "all" in extras or e in extras:
                _log.info("including extra %s", e)
                earr = edeps[e]
                assert isinstance(earr, tomlkit.items.Array)
                lines += array_lines(earr)

    return list(parse_requirements(lines, file))


def array_lines(tbl: tomlkit.items.Array):
    for d in tbl._value:
        if not d.value:
            continue

        text = str(d.value)
        if d.comment:
            text += str(d.comment)
        yield text


def load_conda_yml(file: Path) -> list[ParsedReq]:
    _log.info("loading Conda environment file %s", file)
    with file.open("rt") as yf:
        data = yaml.safe_load(yf)
    deps: list[ParsedReq] = []
    for dep in data["dependencies"]:
        name, ver = parse_conda_spec(dep)
        deps.append(ParsedReq(file, name, conda=(name, ver)))
    return deps


def load_req_txt(file: Path) -> list[ParsedReq]:
    _log.info("loading requirements file %s", file)
    text = file.read_text()
    return list(parse_requirements(text, file))


def parse_requirements(text: str | list[str], path: Path) -> Generator[ParsedReq, None, None]:
    if isinstance(text, str):
        text = text.splitlines()
    for line in text:
        line = line.strip()
        req = None

        # look for include line
        r_m = re.match(r"^\s*-r\s+(.*)", line)
        if r_m:
            fn = r_m.group(1)
            yield from load_req_txt(path.parent / fn)
            continue

        # look for Conda comment
        cc_m = re.match(r"(.*?)\s*#\s*conda:(.*)", line)
        c_str = None
        if cc_m:
            line = cc_m.group(1)
            c_str = cc_m.group(2).strip()

        # remove comments
        line = re.sub("#.*", "", line).strip()

        name = None
        if line:
            req = Requirement(line)
            name = req.name

        if req or c_str:
            pip = False
            conda = None
            if c_str == "@pip":
                pip = True
            elif c_str is not None:
                conda = parse_conda_spec(c_str)
                if name is None:
                    name = conda[0]
            assert name is not None, "cannot find dependency name"
            res = ParsedReq(path, name, req, conda, pip)
            _log.debug("%s: found %s", path, res)
            yield res


def parse_conda_spec(text: str) -> tuple[str, Optional[str]]:
    m = re.match(r"\s*(.*?)\s*([><~=]=?.*)", text)
    if m:
        return m.group(1), m.group(2)
    else:
        return text.strip(), None


def find_conda_executable():
    if options["--micromamba"]:
        path = shutil.which("micromamba")
        if not path:
            _log.error("cannot find micromamba")
            sys.exit(3)
        return path

    path = shutil.which("mamba")
    if path is None:
        path = shutil.which("conda")
    if path is None:
        _log.error("cannot find mamba or conda")
        sys.exit(3)

    return path


def make_env_object(specs: list[ParsedReq], python: Optional[str] = None) -> dict[str, Any]:
    _log.info("creating environment spec for %d requirements", len(specs))
    deps: list[str | dict[str, list[str]]] = []
    pip_deps: list[str] = []
    if python is None:
        python = options["--python-version"]
    if python:
        deps.append(f"python ={python}")

    deps.append("pip")
    for spec in specs:
        if spec.force_pip:
            pip_deps.append(str(spec.requirement))
        else:
            cs = spec.conda_spec()
            assert cs is not None
            deps.append(cs)

    if options["--mkl"]:
        deps += MKL_DEP

    if pip_deps:
        deps.append({"pip": pip_deps})

    return {"channels": CONDA_CHANNELS, "dependencies": deps}


if __name__ == "__main__":
    main()
