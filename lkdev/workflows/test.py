from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Any, Literal, Optional

from ..ghactions import GHJob, GHStep, script
from ._common import PACKAGES, step_checkout

CODECOV_TOKEN = "5cdb6ef4-e80b-44ce-b88d-1402e4dfb781"
META_PYTHON = "3.11"
PYTHONS = ["3.10", "3.11", "3.12"]
PLATFORMS = ["ubuntu-latest", "macos-latest", "windows-latest"]
VANILLA_PLATFORMS = ["ubuntu-latest", "macos-latest"]
FILTER_PATHS = [
    "lenskit/**.py",
    "**pyproject.toml",
    "requirements*.txt",
    "data/**",
    "scripts/coverage-log.tcl",
    ".github/workflows/test.yml",
]


def workflow():
    jobs = {}
    jobs.update(jobs_test_matrix())
    jobs["results"] = jobs_result(list(jobs.keys()))
    return {
        "name": "Automatic Tests",
        "on": {
            "push": {"branches": ["main"], "paths": list(FILTER_PATHS)},
            "pull_request": {"paths": list(FILTER_PATHS)},
        },
        "concurrency": {
            "group": "test-${{github.ref}}",
            "cancel-in-progress": True,
        },
        "jobs": jobs,
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


def steps_setup_conda(options: JobOptions) -> list[GHStep]:
    ctool = ["python -m lkdev.conda", "-o", "ci-environment.yml"]
    if options.extras:
        for e in options.extras:
            ctool += ["-e", e]
    else:
        ctool += ["-e", "all"]
    ctool += options.req_files + [f"{pkg}/pyproject.toml" for pkg in options.required_packages]
    conda_prep = " ".join(ctool)

    pip = ["pip", "install", "--no-deps"]
    pip += [f"-e {pkg}" for pkg in options.required_packages]

    return [
        {
            "name": "🐍 Setup bootstrap Python",
            "uses": "actions/setup-python@v5",
            "with": {"python-version": META_PYTHON},
        },
        {
            "name": "👢 Generate Conda environment file",
            "run": script(f"""
                pip install -e .
                {conda_prep}
            """),
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
                python utils/fix-coverage-paths.py
                coverage xml
                coverage report
            """),
        },
        {
            "name": "📤 Upload test results",
            "uses": "actions/upload-artifact@v4",
            "if": "always()",
            "with": {
                "name": options.test_artifact_name,
                "path": script("""
                    test*.log
                    .coverage
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


def jobs_test_matrix() -> dict[str, GHJob]:
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
                matrix={"python": PYTHONS, "platform": VANILLA_PLATFORMS},
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


def jobs_result(deps: list[str]) -> GHJob:
    return {
        "name": "Test suite results",
        "runs-on": "ubuntu-latest",
        "needs": deps,
        "steps": [
            step_checkout(),
            {
                "name": "Add upstream remote",
                "run": script("""
                    git remote add upstream https://github.com/lenskit/lkpy.git
                    git fetch upstream
                """),
            },
            {
                "name": "🐍 Set up Python",
                "uses": "actions/setup-python@v5",
                "with": {"python-version": META_PYTHON},
            },
            {
                "name": "📦 Install reporting packages",
                "run": "python -m pip install -r requirements-reporting.txt",
            },
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
            # inspired by https://hynek.me/articles/ditch-codecov-python/
            {
                "name": "⛙ Merge and report",
                "run": script("""
                    coverage combine test-logs/*/.coverage
                    coverage xml
                    coverage html -d lenskit-coverage
                    coverage report --format=markdown >coverage.md
                """),
            },
            {
                "name": "Analyze diff coverage",
                "if": "github.event_name == 'pull_request'",
                "run": script("""
                    diff-cover --json-report diff-cover.json --markdown-report diff-cover.md \\
                        coverage.xml |tee diff-cover.txt
                """),
            },
            {
                "name": "± Measure and report coverage",
                "run": script("""
                    echo $PR_NUMBER > ./lenskit-coverage/pr-number
                    tclsh ./utils/coverage-log.tcl
                    cat lenskit-coverage/report.md >$GITHUB_STEP_SUMMARY
                """),
                "env": {"PR_NUMBER": "${{ github.event.number }}"},
            },
            {
                "name": "📤 Upload coverage report",
                "uses": "actions/upload-artifact@v4",
                "if": "always()",
                "with": {
                    "name": "coverage-report",
                    "path": "lenskit-coverage/",
                },
            },
            {
                "name": "🚫 Fail if coverage is too low",
                "run": "coverage report --fail-under=90",
            },
        ],
    }
