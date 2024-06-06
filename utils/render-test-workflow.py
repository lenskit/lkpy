# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Render workflows and project template infrastructure.

Usage:
    render-test-workflow.py [-v] [-o FILE]

Options:
    -v, --verbose           verbose logging
    -o FILE, --output=FILE  write to FILE
"""

# pyright: strict
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Literal, NotRequired, Optional, TypedDict

import yaml
from docopt import docopt

_log = logging.getLogger("render-workflows")
CODECOV_TOKEN = "5cdb6ef4-e80b-44ce-b88d-1402e4dfb781"
PYTHONS = ["3.10", "3.11"]
BASIC_PLATFORMS = ["ubuntu-latest", "macos-latest", "windows-latest"]
ALL_PLATFORMS = BASIC_PLATFORMS + ["macos-13"]
WORKFLOW_HEADER = {
    "name": "Test Suite",
    "on": {
        "push": {
            "branches": ["main"],
        },
        "pull_request": {},
    },
    "concurrency": {
        "group": "test-${{github.ref}}",
        "cancel-in-progress": True,
    },
}


@dataclass
class JobOptions:
    key: str
    name: str
    env: Literal["conda", "vanilla"] = "vanilla"
    runs_on: Optional[str] = None
    python: Optional[str] = None
    matrix: Optional[dict[str, list[str]]] = None
    extras: Optional[list[str]] = None
    pip_args: Optional[list[str]] = None
    req_file: str = "test-requirements.txt"
    test_args: Optional[list[str]] = None
    test_env: Optional[dict[str, str | int]] = None
    packages: list[str] = field(default_factory=lambda: ["lenskit"])

    @property
    def test_artifact_name(self) -> str:
        name = f"test-{self.key}"
        if self.matrix:
            if "platform" in self.matrix:
                name += "-${{matrix.platform}}"
            if "python" in self.matrix:
                name += "-py${{matrix.python}}"
            for key in self.matrix:
                if key not in ["platform", "python"]:
                    name += "-${{matrix." + key + "}}"
        return name

    @property
    def vm_platform(self) -> str:
        if self.runs_on:
            return self.runs_on
        elif self.matrix and "platform" in self.matrix:
            return "${{matrix.platform}}"
        else:
            return "ubuntu-latest"

    @property
    def python_version(self) -> str:
        if self.matrix and "python" in self.matrix:
            return "${{matrix.python}}"
        elif self.python:
            return self.python
        else:
            return PYTHONS[0]

    @property
    def required_packages(self) -> list[str]:
        if "lenskit" not in self.packages:
            return ["lenskit"] + self.packages
        else:
            return self.packages


class script:
    def __init__(self, source: str):
        self.source = dedent(source).strip() + "\n"

    @staticmethod
    def presenter(dumper: yaml.Dumper, script: script):
        return dumper.represent_scalar("tag:yaml.org,2002:str", script.source, style="|")  # type: ignore

    @classmethod
    def command(cls, args: list[str]):
        return cls(" ".join(args))


yaml.add_representer(script, script.presenter)

GHStep = TypedDict(
    "GHStep",
    {
        "id": NotRequired[str],
        "name": NotRequired[str],
        "uses": NotRequired[str],
        "run": NotRequired[str | script],
        "shell": NotRequired["str"],
        "with": NotRequired[dict[str, str | int | bool | script]],
        "env": NotRequired[dict[str, str | int]],
    },
)

GHJob = TypedDict(
    "GHJob",
    {
        "name": str,
        "runs-on": str,
        "timeout-minutes": NotRequired[int],
        "strategy": NotRequired[dict[str, Any]],
        "defaults": NotRequired[dict[str, Any]],
        "needs": NotRequired[list[str]],
        "steps": NotRequired[list[GHStep]],
    },
)


def job_strategy(options: JobOptions) -> dict[str, Any]:
    if options.matrix:
        return {
            "strategy": {
                "fail-fast": False,
                "matrix": {k: v[:] for (k, v) in options.matrix.items()},
            }
        }
    else:
        return {}


def step_checkout(options: Optional[JobOptions] = None) -> GHStep:
    return {
        "name": "🛒 Checkout",
        "uses": "actions/checkout@v4",
        "with": {"fetch-depth": 0},
    }


def steps_setup_conda(options: JobOptions) -> list[GHStep]:
    ctool = ["pipx run", "./utils/conda-tool.py", "--env", "-o", "ci-environment.yml"]
    if options.extras:
        for e in options.extras:
            ctool += ["-e", e]
    else:
        ctool += ["-e", "all"]
    ctool += [options.req_file] + [f"{pkg}/pyproject.toml" for pkg in options.required_packages]

    pip = ["pip", "install", "--no-deps"]
    pip += [f"./{pkg}" for pkg in options.required_packages]

    return [
        {
            "name": "👢 Generate Conda environment file",
            "run": script.command(ctool),
        },
        {
            "id": "setup",
            "name": "📦 Set up Conda environment",
            "uses": "mamba-org/setup-micromamba@v1",
            "with": {
                "environment-file": "ci-environment.yml",
                "environment-name": "lkpy",
                "cache-environment": True,
                "init-shell": "bash",
            },
        },
        {"name": "🍱 Install LensKit packages", "run": script.command(pip)},
    ]


def steps_setup_vanilla(options: JobOptions) -> list[GHStep]:
    pip = ["uv pip install", "--python", "$PYTHON", "-r", options.req_file]
    pip += [f"./{pkg}" for pkg in options.required_packages]
    if options.pip_args:
        pip += options.pip_args

    return [
        {
            "name": "🐍 Set up Python",
            "id": "install-python",
            "uses": "actions/setup-python@v5",
            "with": {
                "python-version": options.python_version,
            },
        },
        {
            "name": "🕶️ Set up uv",
            "run": script("pip install -U 'uv>=0.1.15'"),
        },
        {
            "name": "📦 Set up Python dependencies",
            "id": "install-deps",
            "run": script.command(pip),
            "shell": "bash",
            "env": {
                "PYTHON": "${{steps.install-python.outputs.python-path}}",
                "UV_EXTRA_INDEX_URL": "https://download.pytorch.org/whl/cpu",
                "UV_INDEX_STRATEGY": "unsafe-first-match",
            },
        },
    ]


def steps_inspect(options: JobOptions) -> list[GHStep]:
    return [
        {
            "name": "🔍 Inspect environment",
            "run": script("""
                which -a python
                python -V
                numba -s
            """),
        }
    ]


def steps_test(options: JobOptions) -> list[GHStep]:
    test_cmd = [
        "python",
        "-m",
        "pytest",
        "--cov=lenskit",
        "--verbose",
        "--log-file=test.log",
        "--durations=25",
    ]
    if options.test_args:
        test_cmd += options.test_args
    test_cmd += [f"{pkg}/tests" for pkg in options.packages]
    test: GHStep = {
        "name": "🏃🏻‍➡️ Test LKPY",
        "run": script.command(test_cmd),
    }
    if options.test_env:
        test["env"] = options.test_env
    return [test] + steps_coverage(options)


def steps_coverage(options: JobOptions) -> list[GHStep]:
    return [
        {
            "name": "📐 Coverage results",
            "run": "coverage xml",
        },
        {
            "name": "📤 Upload test results",
            "uses": "actions/upload-artifact@v3",
            "with": {
                "name": options.test_artifact_name,
                "path": script("""
                    test*.log
                    coverage.xml
                """),
            },
        },
    ]


def test_job_steps(options: JobOptions) -> list[GHStep]:
    steps = [
        step_checkout(options),
    ]
    if options.env == "conda":
        steps += steps_setup_conda(options)
    else:
        steps += steps_setup_vanilla(options)
    steps += steps_inspect(options)
    steps += steps_test(options)
    return steps


def test_job(options: JobOptions) -> GHJob:
    job: GHJob = {
        "name": options.name,
        "runs-on": options.vm_platform,
        "timeout-minutes": 30,
    }
    if options.env == "conda":
        job["defaults"] = {
            "run": {
                "shell": "bash -el {0}",
            },
        }

    job.update(job_strategy(options))  # type: ignore
    job["steps"] = test_job_steps(options)
    return job


def test_demo_job() -> GHJob:
    opts = JobOptions(
        "examples",
        "Demos, examples, and docs",
        env="conda",
        req_file="dev-requirements.txt",
        packages=["lenskit-funksvd"],
    )
    cov = "--cov=lenskit/lenskit --cov=lenskit-funksvd/lenskit"
    return {
        "name": opts.name,
        "runs-on": opts.vm_platform,
        "defaults": {"run": {"shell": "bash -el {0}"}},
        "steps": [step_checkout(opts)]
        + steps_setup_conda(opts)
        + [
            {
                "name": "Cache ML data",
                "uses": "actions/cache@v2",
                "with": {
                    "path": script("""
                        data
                        !data/*.zip
                    """),
                    "key": "test-mldata-000",
                },
            },
            {
                "name": "Download ML data",
                "run": script("""
                    python -m lenskit.datasets.fetch ml-100k
                    python -m lenskit.datasets.fetch ml-1m
                    python -m lenskit.datasets.fetch ml-10m
                    python -m lenskit.datasets.fetch ml-20m
                """),
            },
            {
                "name": "Install for testing",
                "run": script("""
                    pip install --no-deps ./lenskit ./lenskit-funksvd
                """),
            },
            {
                "name": "Run Eval Tests",
                "run": script(f"""
                    python -m pytest {cov} -m eval --log-file test-eval.log
                    python -m pytest {cov} --cov-append -m realdata --log-file test-realdata.log
                """),
            },
            {
                "name": "Validate doc notebooks",
                "run": script(f"""
                    cp docs/*.ipynb data
                    python -m pytest {cov} --cov-append --nbval-lax data --log-file test-docs.log
                """),
            },
        ]
        + steps_coverage(opts),
    }


def test_jobs() -> dict[str, GHJob]:
    return {
        "conda": test_job(
            JobOptions(
                "conda",
                "Conda Python ${{matrix.python}} on ${{matrix.platform}}",
                matrix={"python": PYTHONS, "platform": ALL_PLATFORMS},
                env="conda",
            )
        ),
        "vanilla": test_job(
            JobOptions(
                "vanilla",
                "Vanilla Python ${{matrix.python}} on ${{matrix.platform}}",
                matrix={"python": PYTHONS, "platform": BASIC_PLATFORMS},
            )
        ),
        "nojit": test_job(
            JobOptions(
                "nojit",
                "Non-JIT test coverage",
                test_env={"NUMBA_DISABLE_JIT": 1, "PYTORCH_JIT": 0},
                test_args=["-m", '"not slow"'],
            )
        ),
        "mindep": test_job(
            JobOptions(
                "mindep", "Minimal dependency tests", pip_args=["--resolution=lowest-direct"]
            )
        ),
        "examples": test_demo_job(),
        "funksvd": test_job(
            JobOptions(
                "funksvd",
                "FunkSVD tests on Python ${{matrix.python}}",
                packages=["lenskit-funksvd"],
                matrix={"python": PYTHONS},
                env="conda",
            )
        ),
    }


def result_job(deps: list[str]) -> GHJob:
    return {
        "name": "Test suite results",
        "runs-on": "ubuntu-latest",
        "needs": deps,
        "steps": [
            step_checkout(),
            {
                "name": "📥 Download test artifacts",
                "uses": "actions/download-artifact@v3",
                "with": {
                    "path": "test-logs",
                },
            },
            {
                "name": "📋 List log files",
                "run": "ls -lR test-logs",
            },
            {
                "name": "✅ Upload coverage",
                "uses": "codecov/codecov-action@v3",
                "with": {
                    "directory": "test-logs/",
                },
                "env": {"CODECOV_TOKEN": CODECOV_TOKEN},
            },
        ],
    }


def main(options: dict[str, str | int | bool | None]):
    init_logging(options)

    jobs: dict[str, GHJob] = test_jobs()
    jobs["results"] = result_job(list(jobs.keys()))

    workflow = dict(WORKFLOW_HEADER) | {"jobs": jobs}

    if options["--output"]:
        _log.info("writing %s", options["--output"])
        with open(options["--output"], "wt") as wf:
            yaml.dump(workflow, wf, allow_unicode=True, sort_keys=False)
    else:
        yaml.dump(workflow, sys.stdout, allow_unicode=True, sort_keys=False)


def init_logging(options: dict[str, str | int | bool | None]):
    level = logging.DEBUG if options["--verbose"] else logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)


if __name__ == "__main__":
    options = docopt(__doc__)
    main(options)
