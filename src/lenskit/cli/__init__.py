# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os
import sys
import warnings
from importlib.metadata import entry_points
from pathlib import Path

import click
import numpy as np

from lenskit import __version__, configure
from lenskit.config import locate_configuration_root
from lenskit.logging import LoggingConfig, console, get_logger

__all__ = ["lenskit", "main", "version"]
_log = get_logger(__name__)


def main():
    """
    Run the main LensKit CLI.  This just delegates to :fun:`lenskit`, but pretty-prints errors.
    """
    np.set_printoptions(threshold=20)
    try:
        ec = lenskit.main(standalone_mode=False)
    except click.ClickException as e:
        _log.error("CLI error, terminating: %s", e)
        sys.exit(2)
    except Exception as e:
        _log.error("LensKit command failed", exc_info=e)
        sys.exit(3)

    if isinstance(ec, int):
        sys.exit(ec)


@click.group("lenskit")
@click.option("-v", "--verbose", "verbosity", count=True, help="Enable verbose logging output")
@click.option("--skip-log-setup", is_flag=True, hidden=True, envvar="LK_SKIP_LOG_SETUP")
@click.option(
    "-R",
    "--project-root",
    type=Path,
    metavar="DIR",
    help="Look for project root in DIR for configuration files.",
)
def lenskit(verbosity: int, project_root: Path | None, skip_log_setup: bool = False):
    """
    Data and pipeline operations with LensKit.
    """

    # this code is run before any other command logic, so we can do global setup
    if not skip_log_setup:
        lc = LoggingConfig()
        if verbosity:
            lc.set_verbose(verbosity)
        lc.apply()

    if project_root is None:
        if pr := os.environ.get("LK_PROJECT_ROOT"):
            project_root = Path(pr)
        else:
            project_root = locate_configuration_root()

    configure(cfg_dir=project_root)


@lenskit.command("version")
def version():
    """
    Print LensKit version info.
    """
    console.print(f"LensKit version [bold cyan]{__version__}[/bold cyan].")


cli_plugins = entry_points(group="lenskit.cli.plugins")
for plugin in cli_plugins:
    try:
        cmd = plugin.load()
    except ImportError as e:
        warnings.warn(f"cannot load CLI {plugin.name}: {e}", ImportWarning)

    lenskit.add_command(cmd)
