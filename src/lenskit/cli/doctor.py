import os
import platform
import re
import sys
from dataclasses import dataclass
from importlib.metadata import distributions, version
from pathlib import Path

import click
import threadpoolctl
from humanize import naturalsize
from rich.console import Console, ConsoleOptions, group
from rich.padding import Padding
from rich.table import Table

from lenskit import __version__
from lenskit.logging import get_logger, stdout_console
from lenskit.parallel import ensure_parallel_init
from lenskit.parallel.ray import RAY_AVAILABLE

_log = get_logger(__name__)
_gh_out: Path | None = None


@click.command("doctor")
@click.option(
    "--github-output",
    "gh_output",
    envvar="GITHUB_OUTPUT",
    type=Path,
    help="Path to GitHub Actions output file.",
)
@click.option("--packages/--no-packages", default=True, help="List installed packages.")
def doctor(gh_output: Path | None, packages: bool):
    """
    Inspect installed LensKit version and environment.
    """
    global _gh_out
    _gh_out = gh_output
    ensure_parallel_init()
    console = stdout_console()
    console.print(inspect_version())
    console.print(inspect_platform())
    console.print(inspect_compute())
    if RAY_AVAILABLE:
        console.print(inspect_ray())
    console.print(inspect_env())
    if packages:
        console.print(inspect_packages())


@dataclass
class kvp:
    name: str
    value: str | int | bool | float | None
    level: int = 1
    value_style: str = "cyan"

    def __rich_console__(self, console: Console, options: ConsoleOptions):
        text = ""
        if self.level == 1:
            text += f"[bold]{self.name}[/bold]: "
        else:
            text += "  " * (self.level - 1)
            text += self.name + ": "
        text += f"[{self.value_style}]"
        text += str(self.value)
        text += f"[/{self.value_style}]"
        return [text]


@group()
def inspect_version():
    dist_ver = version("lenskit")
    if _gh_out:
        with _gh_out.open("at") as ghf:
            print(f"lenskit_version={dist_ver}", file=ghf)

    yield f"[bold]LensKit version[/bold] [cyan]{dist_ver}[/cyan]"
    if str(dist_ver) != __version__:
        yield f"   [yellow]Version mismatch, internal package version is {__version__}[/yellow]"


@group()
def inspect_platform():
    yield kvp("Python version", platform.python_version())
    yield kvp("Platform", platform.platform(), level=2)
    yield kvp("Location", sys.executable, level=2)


@group()
def inspect_compute():
    import numpy as np
    import torch

    yield ""
    yield kvp("NumPy version", np.__version__)
    yield kvp("PyTorch version", torch.__version__)
    if _gh_out:
        with _gh_out.open("at") as ghf:
            print(f"numpy_version={np.__version__}", file=ghf)
            print(f"pytorch_version={torch.__version__}", file=ghf)

    yield "[bold]PyTorch backends[/bold]:"
    yield kvp("cpu", torch.backends.cpu.get_cpu_capability(), level=2)
    for mod in [torch.cuda, torch.backends.mkl, torch.backends.mps]:
        if mod.is_available():
            stat = "available"
        elif hasattr(mod, "is_built") and mod.is_built():
            stat = "unavailable"
        else:
            stat = "absent"
        name = mod.__name__.split(".")[-1]
        yield kvp(name, stat, level=2)

    yield ""
    yield "[bold]Threading layers[/bold]:"
    for i, pool in enumerate(threadpoolctl.threadpool_info(), 1):
        yield f"  Backend {i}:"
        for k, v in pool.items():
            yield kvp(k, v, level=3)


@group()
def inspect_ray():
    import ray

    yield ""
    yield "[bold]Ray cluster ([yellow]experimental[/yellow])[/bold]:"

    try:
        ray.init("auto", configure_logging=False)
    except ConnectionError:
        yield "  Installed but inactive"
    else:
        yield "  Resources:"
        for name, val in ray.cluster_resources().items():
            if name.startswith("node:"):
                continue
            if name.endswith("memory"):
                val = naturalsize(val)
            yield kvp(name, val, level=3)


@group()
def inspect_env():
    yield ""
    yield "[bold]Relevant environment variables[/bold]:"
    for k, v in os.environ.items():
        if re.match(r"^(LK_|OMP_|NUMBA_|MKL_|TORCH_|PY)", k):
            yield kvp(k, v, level=2)

    yield ""
    yield "[bold]Python search paths[/bold]:"
    for path in sys.path:
        yield f"- {path}"

    yield ""
    yield "[bold]Executable search paths[/bold]:"
    paths = os.environ["PATH"].split(os.pathsep)
    for path in paths:
        yield f"- {path}"


def inspect_packages():
    dists = sorted(distributions(), key=lambda d: d.name)

    n = len(dists)
    table = Table(title=f"Installed Packages ({n})")
    table.add_column("Package")
    table.add_column("Version", justify="right")

    for dist in dists:
        table.add_row(dist.name, dist.version)

    return Padding(table, (1, 0, 0, 2))
