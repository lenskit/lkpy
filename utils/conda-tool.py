#!/usr/bin/env python3
"""
Generate and manage Conda environments.

Usage:
    conda-tool.py [options] --env [-o OUTFILE] REQFILE...
    conda-tool.py [options] --env --all

Options:
    -v, --verbose   enable verbose logging
    -o FILE, --output=FILE
                    write output to FILE
    -p VER, --python-version=VER
                    use Python version VER
    --mkl           enable MKL BLAS
    --cuda          enable CUDA PyTorch
    --env           write a Conda environment specification
"""

# /// script
# dependencies = ["tomlkit>=0.12", "pyyaml==6.*", "packaging>=24.0", "docopt>=0.6"]
# ///

import logging
import re
import sys
from pathlib import Path
from typing import Any, Generator, Iterable, NamedTuple, Optional

import tomlkit
import tomlkit.items
import yaml
from docopt import docopt
from packaging.requirements import Requirement

_log = logging.getLogger("conda-tool")

CONDA_CHANNELS = ["pytorch", "conda-forge", "nodefaults"]
MKL_DEP = ["libblas=*=*mkl*"]
ALL_PYTHONS = ["3.10", "3.11"]
ALL_VARIANTS = {
    "ci": ["pyproject.toml", "test-requirements.txt"],
    "dev": ["pyproject.toml", "dev-requirements.txt"],
    "doc": ["pyproject.toml", "doc-requirements.txt"],
}


class ParsedReq(NamedTuple):
    source: Path
    requirement: Optional[Requirement]
    conda: Optional[tuple[str, Optional[str]]]
    force_pip: bool

    def conda_spec(self):
        name = None
        ver = None
        if self.requirement:
            name = self.requirement.name
        if self.conda:
            name, ver = self.conda

        if self.requirement:
            pip_spec = self.requirement.specifier
            if ver is None and pip_spec is not None:
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
    init_logging()

    if options["--env"]:
        if options["--all"]:
            for py in ALL_PYTHONS:
                for var, files in ALL_VARIANTS.items():
                    fn = f"envs/lenskit-py{py}-{var}.yaml"
                    reqs = list(load_reqfiles(files))
                    make_env_file(reqs, fn, python=py)
        else:
            make_env_file(list(load_reqfiles(options["REQFILE"])), options["--output"])


def init_logging():
    level = logging.DEBUG if options["--verbose"] else logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)


def load_reqfiles(files: Iterable[Path | str]):
    for file in files:
        yield from load_requirements(Path(file))


def load_requirements(file: Path) -> list[ParsedReq]:
    if file.name == "pyproject.toml":
        return load_req_toml(file)
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
    dtext = deps.as_string()
    lines = [
        re.sub(r'^\s+"(.*)",', "\\1", line)
        for line in dtext.splitlines()
        if re.match(r'^\s*"', line)
    ]
    return list(parse_requirements(lines, file))


def load_req_txt(file: Path) -> list[ParsedReq]:
    _log.info("loading requirements file %s", file)
    text = file.read_text()
    return list(parse_requirements(text, file))


def parse_requirements(text: str | list[str], path: Path) -> Generator[ParsedReq, None, None]:
    if isinstance(text, str):
        text = text.splitlines()
    for line in text:
        line = line.strip()

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

        if line:
            req = Requirement(line)

        if req or c_str:
            pip = False
            conda = None
            if c_str == "@pip":
                pip = True
            elif c_str is not None:
                conda = parse_conda_spec(c_str)
            res = ParsedReq(path, req, conda, pip)
            _log.debug("%s: found %s", path, res)
            yield res


def parse_conda_spec(text: str) -> tuple[str, Optional[str]]:
    m = re.match(r"\s*(.*?)\s*([><~=]=?.*)", text)
    if m:
        return m.group(1), m.group(2)
    else:
        return text.strip(), None


def make_env_file(specs: list[ParsedReq], outfile: str | None, python=None):
    env = make_env_object(specs, python)

    if outfile:
        _log.info("writing %s", outfile)
        with open(outfile, "wt") as outf:
            yaml.safe_dump(env, outf, indent=2)
    else:
        yaml.safe_dump(env, sys.stdout, indent=2)


def make_env_object(specs: list[ParsedReq], python=None) -> dict[str, Any]:
    _log.info("creating environment spec for %d requirements", len(specs))
    deps = []
    pip_deps = []
    if python is None:
        python = options["--python-version"]
    if python:
        deps.append(f"python ={python}")

    for spec in specs:
        if spec.force_pip:
            pip_deps.append(str(spec.requirement))
        else:
            deps.append(spec.conda_spec())

    if options["--mkl"]:
        deps.append(MKL_DEP)

    if pip_deps:
        deps.append("pip")
        deps.append({"pip": pip_deps})

    return {"channels": CONDA_CHANNELS, "dependencies": deps}


if __name__ == "__main__":
    options = docopt(__doc__)
    main()
