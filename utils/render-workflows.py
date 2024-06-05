"""
Render workflows and project template infrastructure.

Usage:
    render-workflows.py [-v]

Options:
    -v, --verbose   verbose logging
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Literal, NotRequired, Optional, TypedDict

import yaml
from docopt import docopt

_log = logging.getLogger("render-workflows")
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
    "defaults": {
        "run": {
            "shell": "bash -el {0}",
        },
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
    extras: list[str] = field(default=list)
    pip_args: Optional[list[str]] = None
    req_file: str = "test-requirements.txt"
    test_args: Optional[list[str]] = None
    test_env: Optional[dict[str, str | int]] = None

    @property
    def test_artifact_name(self):
        name = f"test-{self.key}"
        if self.matrix:
            if "platform" in self.matrix:
                name += "-${{matrix.platform}}"
            if "python" in self.matrix:
                name += "-py${{matrix.python}}"
            for key in self.matrix:
                if key not in ["platform", "python"]:
                    name += "-${{matrix." + key + "}}"

    @property
    def vm_platform(self) -> str:
        if self.runs_on:
            return self.runs_on
        elif self.matrix and "platform" in self.matrix:
            return self.matrix["platform"][0]
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


class script:
    def __init__(self, source: str):
        self.source = dedent(source)

    @staticmethod
    def presenter(dumper, script: script):
        return dumper.represent_scaler("tag:yaml.org,2002:str", script.source, style="|")

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
        "with": NotRequired[dict[str, str | int | bool]],
        "env": NotRequired[dict[str, str | int]],
    },
)


def job_strategy(options: JobOptions):
    if options.matrix:
        return {"strategy": {"fail-fast": False, "matrix": options.matrix}}
    else:
        return {}


def step_checkout(options: Optional[JobOptions] = None) -> GHStep:
    return {
        "name": "🛒 Checkout",
        "uses": "actions/checkout@v4",
        "with": {"fetch-depth": 0},
    }


def step_setup_conda(options: JobOptions) -> list[GHStep]:
    extra_args = []
    for e in options.extras:
        extra_args += ["-e", e]
    if not extra_args:
        extra_args = ["-e", "all"]

    return [
        {
            "name": "👢 Generate Conda environment file",
            "run": f"pipx run ./utils/conda-tool.py --env -o ci-environment.yml {extra_args} pyproject.toml dev-requirements.txt",
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
    ]


def step_setup_vailla(options: JobOptions):
    pip = ["uv pip install", "--python", "$PYTHON", "-r", options.req_file, "-e", "."]
    if options.pip_args:
        pip += options.pip_args
    return [
        {
            "name": "🐍 Set up Python",
            "uses": "actions/setup-python@v5",
            "id": "pyinstall",
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
            "env": {
                "PYTHON": "${{steps.install-python.outputs.python-path}}",
                "UV_EXTRA_INDEX_URL": "https://download.pytorch.org/whl/cpu",
                "UV_INDEX_STRATEGY": "unsafe-first-match",
            },
        },
    ]


def steps_inspect(options: JobOptions):
    return [
        {
            "name": "🔍 Inspect environment",
            "run": script("""
                python -V
                numba -s
            """),
        }
    ]


def steps_test(options: JobOptions):
    if options.test_args:
        test_args = " ".join(options.test_args)
    else:
        test_args = ""
    test = {
        "name": "🏃🏻‍➡️ Test LKPY",
        "run": script(f"""
        python -m pytest --cov=lenskit --verbose --log-file=test.log --durations=25 {test_args}
    """),
    }
    if options.test_env:
        test["env"] = options.test_env
    return [
        test,
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


def test_job_steps(options: JobOptions):
    steps = [
        step_checkout(options),
    ]
    if options.env == "conda":
        steps += step_setup_conda(options)
    else:
        steps += step_setup_vailla(options)
    steps += steps_inspect(options)
    steps += steps_test(options)


def test_job(options: JobOptions):
    job = {
        "name": options.name,
        "runs-on": options.vm_platform,
        "timeout-minutes": 30,
    }
    job.update(job_strategy(options))
    job["steps"] = test_job_steps(options)


def test_jobs():
    return {
        "conda": test_job(
            JobOptions(
                "conda",
                "Conda Python ${{matrix.python}} on ${{matrix.platform}}",
                matrix={"python": PYTHONS, "platform": ALL_PLATFORMS},
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
    }


def result_job(deps: list[str]):
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
            },
        ],
    }


def main(options):
    init_logging(options)
    dir = Path(".github/workflows")

    jobs = test_jobs()
    jobs["results"] = result_job(list(jobs.keys()))

    workflow = dict(WORKFLOW_HEADER) | {"jobs": jobs}


def init_logging(options):
    level = logging.DEBUG if options["--verbose"] else logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)


if __name__ == "__main__":
    options = docopt(__doc__)
    main(options)
