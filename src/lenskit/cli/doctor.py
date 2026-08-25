# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os
import platform
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import distributions, version
from pathlib import Path
from typing import Any

import click
import psutil
import rich
from cpuinfo import get_cpu_info
from humanize import metric, naturalsize
from rich.console import Console, ConsoleOptions, Group, group
from rich.padding import Padding
from rich.table import Table

from lenskit import __version__, _accel, lenskit_config
from lenskit.config import LenskitSettings
from lenskit.logging import get_logger, stdout_console
from lenskit.logging.tasks import measure_power
from lenskit.parallel import ensure_parallel_init
from lenskit.parallel.config import (
    effective_cpu_count,
    get_parallel_config,
    is_free_threaded,
)
from lenskit.parallel.ray import ray_available

_log = get_logger(__name__)
_gh_out: Path | None = None

INDENT_SIZE = 2


class Inspector(ABC):
    """
    Base class for individual inspectors.
    """

    def enabled(self) -> bool:
        "Query whether this inspector is enabled."
        return True

    @abstractmethod
    def header(self) -> Any:
        "Section header for the inspector."
        ...

    def body(self) -> Any:
        "Emit the inspector body."
        return []


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
@click.option("--full", is_flag=True, default=False, help="Show all output.")
def doctor(gh_output: Path | None, packages: bool, paths: bool, full: bool):
    """
    Inspect installed LensKit version and environment.
    """
    global _gh_out
    _gh_out = gh_output
    config = lenskit_config()
    ensure_parallel_init()
    console = stdout_console()

    inspect(VersionInspector(), console)
    inspect(PlatformInspector(), console)
    inspect(SystemInspector(), console)
    inspect(ComputeInspector(), console)
    inspect(TorchInspector(), console)
    inspect(ThreadInspector(), console)
    inspect(RayInspector(), console)
    inspect(ParallelInspector(), console)
    inspect(PowerInspector(config), console)
    inspect(EnvInspector(), console)
    if paths or full:
        inspect(PythonPathInspector(), console)
        inspect(ProgramPathInspector(), console)
    if packages or full:
        inspect(PackageInspector(), console)


def inspect(what: Inspector, console: rich.console.Console):
    if what.enabled():
        console.print(what.header(), highlight=False)
        console.print(indent(what.body()), highlight=False)
        console.print()
    else:
        _log.debug("%s disabled", what.__class__.__name__)


def indent(obj):
    return Padding.indent(obj, INDENT_SIZE)


@dataclass
class kvp:
    name: str
    value: str | int | bool | float | None
    value_style: str = "cyan"

    def __rich_console__(self, console: Console, options: ConsoleOptions):
        text = f"[bold]{self.name}[/bold]: "
        text += f"[{self.value_style}]"
        text += str(self.value)
        text += f"[/{self.value_style}]"
        return [text]


class VersionInspector(Inspector):
    def header(self):
        dist_ver = version("lenskit")
        if _gh_out:
            with _gh_out.open("at") as ghf:
                print(f"lenskit_version={dist_ver}", file=ghf)

        return f"[bold]LensKit version:[/bold] [cyan]{dist_ver}[/cyan]"

    @group()
    def body(self):
        dist_ver = version("lenskit")
        if str(dist_ver) != __version__:
            yield f"[yellow]Version mismatch, internal package version is {__version__}[/yellow]"


class PlatformInspector(Inspector):
    def header(self):
        return kvp("Python version", platform.python_version())

    @group()
    def body(self):
        yield kvp("Platform", platform.platform())
        yield kvp("Location", sys.executable)
        if is_free_threaded(require_active=True):
            yield "[bold][green]Free-threading available[/green][/bold]"
        elif is_free_threaded():
            yield "[bold][yellow]Python is free-threaded but cannot disable GIL[/yellow][/bold]"
        else:
            yield "[yellow]Python GIL enabled[/yellow]"


class SystemInspector(Inspector):
    def header(self):
        return "[bold]System information:[/bold]"

    @group()
    def body(self):
        cpu = get_cpu_info()
        yield kvp("Processor", cpu["brand_raw"])
        if freq := cpu.get("hz_advertised", None):
            yield kvp("CPU Frequency", metric(freq[0], unit="Hz"))

        eff_cpu = effective_cpu_count()
        ncpus = os.cpu_count()
        cpus = f"[bold]{ncpus}[/bold]"
        nphys = psutil.cpu_count(logical=False)
        if nphys != ncpus:
            cpus += f" ({nphys} physical)"
        if ncpus != eff_cpu:
            cpus += f", limited to {ncpus}"
        yield kvp("CPU cores", cpus)

        vmem = psutil.virtual_memory()
        yield kvp(
            "Memory",
            f"[bold]{naturalsize(vmem.total, binary=True)}[/bold]"
            f" ({naturalsize(vmem.available, binary=True)} available)",
        )


class ParallelInspector(Inspector):
    def header(self):
        return "[bold]Parallel configuration:[/bold]"

    @group()
    def body(self):
        pc = get_parallel_config()
        yield kvp("available CPUs", pc.num_cpus)
        yield kvp("batch jobs", pc.num_batch_jobs)
        yield kvp("threads", pc.num_threads)
        yield kvp("backend threads", pc.num_backend_threads)


class ComputeInspector(Inspector):
    def header(self):
        return "[bold]Compute configuration:[/bold]"

    @group()
    def body(self):
        import numpy as np
        import torch

        yield kvp("NumPy version", np.__version__)
        yield kvp("PyTorch version", torch.__version__)
        if _gh_out:
            with _gh_out.open("at") as ghf:
                print(f"numpy_version={np.__version__}", file=ghf)
                print(f"pytorch_version={torch.__version__}", file=ghf)


class TorchInspector(Inspector):
    def header(self):
        return "[bold]PyTorch backends[/bold]:"

    @group()
    def body(self):
        import torch

        try:
            import cupy  # type: ignore

            _log.debug("imported CuPy version %s", cupy.__version__)
        except ImportError:
            _log.debug("CuPy unavailable")
            cupy = None

        yield kvp("cpu", torch.backends.cpu.get_cpu_capability())
        for mod in [torch.cuda, torch.backends.mkl, torch.backends.mps]:
            if mod.is_available():
                stat = "available"
            elif hasattr(mod, "is_built") and mod.is_built():
                stat = "unavailable"
            else:
                stat = "absent"
            name = mod.__name__.split(".")[-1]
            yield kvp(name, stat)

        if torch.cuda.is_available():
            yield ""
            yield "[bold]PyTorch GPUs[/bold]:"
            for dev in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(dev)
                yield f"  [green]cuda:{dev}[/green]: [bold cyan]{props.name}[/bold cyan]"
                yield kvp("capability", f"{props.major}.{props.minor}")
                yield kvp("memory", naturalsize(props.total_memory, binary=True))
                yield kvp("L2 cache", naturalsize(props.L2_cache_size, binary=True))
                yield kvp("MP count", props.multi_processor_count)
                if cupy is not None:
                    cd = cupy.cuda.Device(dev)
                    yield kvp("warp size", cd.attributes["WarpSize"])
                    yield kvp("blocks/MP", cd.attributes["MaxBlocksPerMultiprocessor"])
                    yield kvp("clock rate", metric(cd.attributes["ClockRate"], unit="Hz"))
                    yield kvp("sp/dp ratio", cd.attributes["SingleToDoublePrecisionPerfRatio"])


class ThreadInspector(Inspector):
    def header(self):
        return "[bold]Threading layers[/bold]"

    @group()
    def body(self):
        import threadpoolctl

        yield "Rayon:"
        yield indent(kvp("threads", _accel.thread_count()))
        for i, pool in enumerate(threadpoolctl.threadpool_info(), 1):
            yield f"Backend {i}:"
            yield indent(Group(*[kvp(k, v) for (k, v) in pool.items()]))


class RayInspector(Inspector):
    def enabled(self):
        return ray_available()

    def header(self):
        return "[bold]Ray cluster ([yellow]experimental[/yellow])[/bold]"

    @group()
    def body(self):
        import ray

        try:
            ray.init("auto", configure_logging=False)
        except ConnectionError:
            yield "Installed but inactive"
        except RuntimeError:
            yield "Cannot connect"
        else:
            yield "Resources:"
            for name, val in ray.cluster_resources().items():
                if name.startswith("node:"):
                    continue
                if name.endswith("memory"):
                    val = naturalsize(val)
                yield indent(kvp(name, val))


class PowerInspector(Inspector):  # pragma: nocover
    config: LenskitSettings

    def __init__(self, config: LenskitSettings):
        self.config = config

    def enabled(self):
        return self.config.machines

    def header(self):
        return "[bold]Power Measurement:[/bold]"

    @group()
    def body(self):
        if m := self.config.current_machine:
            yield kvp("Machine", f"[bold][yellow]{self.config.machine}[/yellow][/bold]")

            if "system" in m.power_queries:
                pow = measure_power("system", 60, config=self.config)
                pow_s = metric(pow, "J")
                yield kvp("System power", f"{pow_s} (in last 60s)")
            else:
                yield kvp("System power", "not configured")

            if "cpu" in m.power_queries:
                pow = measure_power("cpu", 60, config=self.config)
                pow_s = metric(pow, "J")
                yield kvp("CPU power", f"{pow_s} (in last 60s)")
            else:
                yield kvp("CPU power", "not configured")

            if "gpu" in m.power_queries:
                pow = measure_power("gpu", 60, config=self.config)
                pow_s = metric(pow, "J")
                yield kvp("GPU power", f"{pow_s} (in last 60s)")
            else:
                yield kvp("GPU power", "not configured")

        elif self.config.machine:
            yield f"[red]Machine [white]{self.config.machine}[/white] is not configured[/red] (see https://lenskit.org/q/power)"

        else:
            yield "[yellow]No machine configured[/yellow] (see https://lenskit.org/q/power)"


class EnvInspector(Inspector):
    def header(self):
        return "[bold]Relevant environment variables:[/bold]"

    @group()
    def body(self):
        for k, v in os.environ.items():
            if re.match(r"^(LK_|OMP_|NUMBA_|MKL_|TORCH_|PY)", k):
                yield kvp(k, v)


class PythonPathInspector(Inspector):
    def header(self):
        return "[bold]Python search paths:[/bold]"

    @group()
    def body(self):
        for path in sys.path:
            yield f"{path}"


class ProgramPathInspector(Inspector):
    def header(self):
        return "[bold]Executable search paths:[/bold]"

    @group()
    def body(self):
        exe_paths = os.environ["PATH"].split(os.pathsep)
        for path in exe_paths:
            yield f"- {path}"


class PackageInspector(Inspector):
    def header(self):
        return ""

    @group()
    def body(self):
        dists = sorted(distributions(), key=lambda d: d.name or "UNNAMED")

        n = len(dists)
        table = Table(title=f"Installed Packages ({n})")
        table.add_column("Package")
        table.add_column("Version", justify="right")

        for dist in dists:
            table.add_row(dist.name, dist.version)

        return Padding(table, (1, 0, 0, 2))
