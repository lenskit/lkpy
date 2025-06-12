# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os
import platform
import re
import sys
from dataclasses import dataclass
from importlib.metadata import distributions, version
from pathlib import Path

import click
import psutil
import threadpoolctl
from cpuinfo import get_cpu_info
from humanize import metric, naturalsize
from rich.console import Console, ConsoleOptions, group
from rich.padding import Padding
from rich.table import Table

from lenskit import __version__, _accel
from lenskit.logging import get_logger, stdout_console
from lenskit.parallel import ensure_parallel_init
from lenskit.parallel.config import effective_cpu_count
from lenskit.parallel.ray import ray_available

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
@click.option("--packages/--no-packages", default=False, help="List installed packages.")
@click.option("--paths/--no-paths", default=False, help="List search paths.")
def doctor(gh_output: Path | None, packages: bool, paths: bool):
    """
    Inspect installed LensKit version and environment.
    """
    global _gh_out
    _gh_out = gh_output
    ensure_parallel_init()
    console = stdout_console()
    console.print(inspect_version(), highlight=False)
    console.print(inspect_platform(), highlight=False)
    console.print(inspect_system(), highlight=False)
    console.print(inspect_compute(), highlight=False)
    if ray_available():
        console.print(inspect_ray(), highlight=False)
    console.print(inspect_env(paths), highlight=False)
    if packages:
        console.print(inspect_packages(), highlight=False)


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

    yield f"[bold]LensKit version:[/bold] [cyan]{dist_ver}[/cyan]"
    if str(dist_ver) != __version__:
        yield f"   [yellow]Version mismatch, internal package version is {__version__}[/yellow]"


@group()
def inspect_platform():
    yield kvp("Python version", platform.python_version())
    yield kvp("Platform", platform.platform(), level=2)
    yield kvp("Location", sys.executable, level=2)


@group()
def inspect_system():
    yield ""
    yield "[bold]System information:[/bold]"

    cpu = get_cpu_info()
    yield kvp("Processor", cpu["brand_raw"], level=2)
    if freq := cpu.get("hz_advertised", None):
        yield kvp("Design CPU Frequency", metric(freq[0], unit="Hz"), level=2)
    if freq := cpu.get("hz_actual", None):
        yield kvp("Current CPU Frequency", metric(freq[0], unit="Hz"), level=2)

    eff_cpu = effective_cpu_count()
    ncpus = os.cpu_count()
    cpus = f"[bold]{ncpus}[/bold]"
    nphys = psutil.cpu_count(logical=False)
    if nphys != ncpus:
        cpus += f" ({nphys} physical)"
    if ncpus != eff_cpu:
        cpus += f", limited to {ncpus}"
    yield kvp("CPU cores", cpus, level=2)

    vmem = psutil.virtual_memory()
    yield kvp(
        "Memory",
        f"[bold]{naturalsize(vmem.total, binary=True)}[/bold]"
        f" ({naturalsize(vmem.available, binary=True)} available)",
        level=2,
    )


@group()
def inspect_compute():
    import numpy as np
    import torch

    try:
        import cupy  # type: ignore

        _log.debug("imported CuPy version %s", cupy.__version__)
    except ImportError:
        _log.debug("CuPy unavailable")
        cupy = None

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

    if torch.cuda.is_available():
        yield ""
        yield "[bold]PyTorch GPUs[/bold]:"
        for dev in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(dev)
            yield "  [green]cuda:{}[/green]: [bold cyan]{}[/bold cyan]".format(dev, props.name)
            yield kvp("capability", f"{props.major}.{props.minor}", level=2)
            yield kvp("memory", naturalsize(props.total_memory), level=2)
            yield kvp("L2 cache", naturalsize(props.L2_cache_size), level=2)
            yield kvp("MP count", props.multi_processor_count, level=2)
            if cupy is not None:
                cd = cupy.cuda.Device(dev)
                yield kvp("warp size", cd.attributes["WarpSize"], level=2)
                yield kvp("blocks/MP", cd.attributes["MaxBlocksPerMultiprocessor"], level=2)
                yield kvp("clock rate", metric(cd.attributes["ClockRate"], unit="Hz"), level=2)
                yield kvp("sp/dp ratio", cd.attributes["SingleToDoublePrecisionPerfRatio"], level=2)

    yield ""
    yield "[bold]Threading layers[/bold]:"
    yield "  Rayon:"
    yield kvp("threads", _accel.thread_count(), level=3)
    for i, pool in enumerate(threadpoolctl.threadpool_info(), 1):
        yield f"  Backend {i}:"
        for k, v in pool.items():
            yield kvp(k, v, level=3)


@group()
def inspect_ray():
    import ray  # type: ignore

    yield ""
    yield "[bold]Ray cluster ([yellow]experimental[/yellow])[/bold]:"

    try:
        ray.init("auto", configure_logging=False)
    except ConnectionError:
        yield "  Installed but inactive"
    except RuntimeError:
        yield "  Cannot connect"
    else:
        yield "  Resources:"
        for name, val in ray.cluster_resources().items():
            if name.startswith("node:"):
                continue
            if name.endswith("memory"):
                val = naturalsize(val)
            yield kvp(name, val, level=3)


@group()
def inspect_env(paths: bool):
    yield ""
    yield "[bold]Relevant environment variables[/bold]:"
    for k, v in os.environ.items():
        if re.match(r"^(LK_|OMP_|NUMBA_|MKL_|TORCH_|PY)", k):
            yield kvp(k, v, level=2)

    if not paths:
        return

    yield ""
    yield "[bold]Python search paths[/bold]:"
    for path in sys.path:
        yield f"- {path}"

    yield ""
    yield "[bold]Executable search paths[/bold]:"
    exe_paths = os.environ["PATH"].split(os.pathsep)
    for path in exe_paths:
        yield f"- {path}"


def inspect_packages():
    dists = sorted(distributions(), key=lambda d: d.name or "UNNAMED")

    n = len(dists)
    table = Table(title=f"Installed Packages ({n})")
    table.add_column("Package")
    table.add_column("Version", justify="right")

    for dist in dists:
        table.add_row(dist.name, dist.version)

    return Padding(table, (1, 0, 0, 2))
