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
from hashlib import blake2b
from textwrap import dedent
from typing import Any, Literal, NotRequired, Optional, TypedDict

import yaml
from docopt import docopt

_log = logging.getLogger("render-workflows")
CODECOV_TOKEN = "5cdb6ef4-e80b-44ce-b88d-1402e4dfb781"
PYTHONS = ["3.10", "3.11", "3.12"]
PLATFORMS = ["ubuntu-latest", "macos-latest", "windows-latest"]
PACKAGES = ["lenskit", "lenskit-funksvd", "lenskit-implicit"]
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
    dep_strategy: Literal["default", "minimum"] = "default"
    req_files: list[str] = field(default_factory=lambda: ["requirements-test.txt"])
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
    ctool += options.req_files + [f"{pkg}/pyproject.toml" for pkg in options.required_packages]

    pip = ["pip", "install", "--no-deps"]
    pip += [f"-e {pkg}" for pkg in options.required_packages]

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
    pip = ["uv pip install", "--python", "$PYTHON"]
    for req in options.req_files:
        pip += ["-r", req]
    pip += [f"-e {pkg}" for pkg in options.required_packages]
    if options.dep_strategy == "minimum":
        pip.append("--resolution=lowest-direct")
    if options.pip_args:
        pip += options.pip_args

    return [
        {
            "name": "🐍 Set up Python",
            "id": "install-python",
            "uses": "actions/setup-python@v5",
            "with": {
                "python-version": options.python_version,
                "cache": "pip",
                "cache-dependency-path": script("""
                    requirements*.txt
                    */pyproject.toml
                """),
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
                python -m lenskit.util.envcheck
            """),
        }
    ]


def steps_mldata(options: JobOptions, datasets: list[str]) -> list[GHStep]:
    ds_str = " ".join(datasets)
    ds_hash = blake2b(ds_str.encode("ascii")).hexdigest()
    return [
        {
            "name": "Cache ML data",
            "uses": "actions/cache@v4",
            "with": {
                "path": script("""
                        data
                        !data/*.zip
                    """),
                "key": f"test-mldata-000-{ds_hash}",
            },
        },
        {
            "name": "Download ML data",
            "run": script(f"""
                python -m lenskit.datasets.fetch {ds_str}
            """),
        },
    ]


def steps_test(options: JobOptions) -> list[GHStep]:
    test_cmd = [
        "python",
        "-m",
        "pytest",
        "--verbose",
        "--log-file=test.log",
        "--durations=25",
    ]
    if options.test_args:
        test_cmd += options.test_args
    test_cmd += [f"--cov={pkg}/lenskit" for pkg in options.packages]
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
            "run": script("""
                coverage xml
                coverage report
            """),
        },
        {
            "name": "📤 Upload test results",
            "uses": "actions/upload-artifact@v4",
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


def test_eval_job() -> GHJob:
    opts = JobOptions(
        "eval-tests",
        "Evaluation-based tests",
        env="conda",
        req_files=["requirements-test.txt"],
        packages=PACKAGES,
    )
    cov = " ".join([f"--cov={pkg}/lenskit" for pkg in PACKAGES])
    return {
        "name": opts.name,
        "runs-on": opts.vm_platform,
        "defaults": {"run": {"shell": "bash -el {0}"}},
        "steps": [step_checkout(opts)]
        + steps_setup_conda(opts)
        + steps_mldata(opts, ["ml-100k", "ml-20m"])
        + [
            {
                "name": "Run Eval Tests",
                "run": script(f"""
                    python -m pytest {cov} -m 'eval or realdata' --log-file test-eval.log */tests
                """),
            },
        ]
        + steps_coverage(opts),
    }


def test_doc_job() -> GHJob:
    opts = JobOptions(
        "examples",
        "Demos, examples, and docs",
        env="conda",
        req_files=["requirements-test.txt", "requirements-demo.txt"],
        packages=PACKAGES,
    )
    cov = " ".join([f"--cov={pkg}/lenskit" for pkg in PACKAGES])
    return {
        "name": opts.name,
        "runs-on": opts.vm_platform,
        "defaults": {"run": {"shell": "bash -el {0}"}},
        "steps": [step_checkout(opts)]
        + steps_setup_conda(opts)
        + steps_mldata(opts, ["ml-100k", "ml-1m", "ml-10m", "ml-20m"])
        + [
            {
                "name": "📕 Validate documentation examples",
                "run": script(f"""
                    python -m pytest {cov} --nbval-lax --log-file test-docs.log docs */lenskit
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
                matrix={"python": PYTHONS, "platform": PLATFORMS},
                env="conda",
            )
        ),
        "vanilla": test_job(
            JobOptions(
                "vanilla",
                "Vanilla Python ${{matrix.python}} on ${{matrix.platform}}",
                matrix={"python": PYTHONS, "platform": PLATFORMS},
            )
        ),
        "nojit": test_job(
            JobOptions(
                "nojit",
                "Non-JIT test coverage",
                packages=["lenskit", "lenskit-funksvd"],
                test_env={"NUMBA_DISABLE_JIT": 1, "PYTORCH_JIT": 0},
                test_args=["-m", "'not slow'"],
            )
        ),
        "mindep": test_job(
            JobOptions(
                "mindep",
                "Minimal dependency tests",
                dep_strategy="minimum",
            )
        ),
        "funksvd": test_job(
            JobOptions(
                "funksvd",
                "FunkSVD tests on Python ${{matrix.python}}",
                packages=["lenskit-funksvd"],
                matrix={"python": PYTHONS},
                env="conda",
            )
        ),
        "funksvd-mindep": test_job(
            JobOptions(
                "mindep-funksvd",
                "Minimal dependency tests for FunkSVD",
                dep_strategy="minimum",
                packages=["lenskit-funksvd"],
            )
        ),
        "implicit": test_job(
            JobOptions(
                "implicit",
                "Implicit bridge tests on Python ${{matrix.python}}",
                packages=["lenskit-implicit"],
                matrix={"python": PYTHONS},
                env="conda",
            )
        ),
        "implicit-mindep": test_job(
            JobOptions(
                "mindep-implicit",
                "Minimal dependency tests for Implicit",
                dep_strategy="minimum",
                packages=["lenskit-implicit"],
            )
        ),
        "hpf": test_job(
            JobOptions(
                "hpf",
                "HPF bridge tests on Python ${{matrix.python}}",
                packages=["lenskit-hpf"],
                matrix={"python": PYTHONS},
                env="vanilla",
            )
        ),
        "eval-tests": test_eval_job(),
        "doc-tests": test_doc_job(),
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
                "uses": "actions/download-artifact@v4",
                "with": {
                    "pattern": "test-*",
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
