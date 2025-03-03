import os
import platform
import re
import sys
from importlib.metadata import distributions, version
from pathlib import Path

import click
import threadpoolctl

from lenskit import __version__

_gh_out: Path | None = None


@click.command("doctor")
@click.option(
    "--github-output",
    "gh_output",
    envvar="GITHUB_OUTPUT",
    type=Path,
    help="Path to GitHub Actions output file.",
)
def doctor(gh_output: Path | None):
    """
    Inspect installed LensKit version and environment.
    """
    global _gh_out
    _gh_out = gh_output
    inspect_version()
    inspect_platform()
    inspect_compute()
    inspect_env()
    inspect_packages()


def inspect_version():
    dist_ver = version("lenskit")
    print("LensKit version:", dist_ver)
    if str(dist_ver) != __version__:
        print("   Version mismatch, internal package version is", __version__)
    if _gh_out:
        with _gh_out.open("at") as ghf:
            print(f"lenskit_version={dist_ver}", file=ghf)


def inspect_platform():
    print("Python version:", platform.python_version())
    print("Platform:", platform.platform())
    print("Python location:", sys.executable)


def inspect_compute():
    import numpy as np
    import torch

    print("\nNumPy version:", np.__version__)
    print("PyTorch version:", torch.__version__)
    if _gh_out:
        with _gh_out.open("at") as ghf:
            print(f"numpy_version={np.__version__}", file=ghf)
            print(f"pytorch_version={torch.__version__}", file=ghf)

    print("PyTorch backends:")
    print("    cpu:", torch.backends.cpu.get_cpu_capability())
    for mod in [torch.cuda, torch.backends.mkl, torch.backends.mps]:
        if mod.is_available():
            stat = "available"
        elif hasattr(mod, "is_built") and mod.is_built():
            stat = "unavailable"
        else:
            stat = "absent"
        name = mod.__name__.split(".")[-1]
        print(f"    {name}: {stat}")

    print("\nThreading layers:")
    for i, pool in enumerate(threadpoolctl.threadpool_info(), 1):
        print(f"    Backend {i}:")
        for k, v in pool.items():
            print("      {}: {}".format(k, v))


def inspect_env():
    print("\nRelevant environment variables:")
    for k, v in os.environ.items():
        if re.match(r"^(LK_|OMP_|NUMBA_|MKL_|TORCH_|PY)", k):
            print(f"    {k}: {v}")
    print("\nPython search paths:")
    for path in sys.path:
        print(f"    {path}")
    print("\nExecutable search paths:")
    paths = os.environ["PATH"].split(os.pathsep)
    for path in paths:
        print(f"    {path}")


def inspect_packages():
    dists = sorted(distributions(), key=lambda d: d.name)
    print("\nInstalled Python packages ({}):".format(len(dists)))
    for dist in dists:
        print("    {:32s}  {:>10s}".format(dist.name, dist.version))
