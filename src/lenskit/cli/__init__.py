# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
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
    except click.ClickException as e:  # pragma: nocover
        _log.error("CLI error, terminating: %s", e)
        sys.exit(2)
    except click.Abort:  # pragma: nocover
        _log.error("Program interrupted")
        sys.exit(3)
    except Exception as e:  # pragma: nocover
        _log.error("LensKit command failed", exc_info=e)
        sys.exit(3)

    if isinstance(ec, int):
        sys.exit(ec)


@click.group("lenskit", invoke_without_command=True)
@click.help_option("-h", "--help")
@click.option("-v", "--verbose", "verbosity", count=True, help="Enable verbose logging output.")
@click.option("--skip-log-setup", is_flag=True, hidden=True, envvar="LK_SKIP_LOG_SETUP")
@click.option("--list-commands", is_flag=True, hidden=True)
@click.option(
    "-R",
    "--project-root",
    type=Path,
    metavar="DIR",
    help="Look for project root in DIR for configuration files.",
)
@click.pass_context
def lenskit(
    ctx: click.Context,
    verbosity: int,
    project_root: Path | None,
    skip_log_setup: bool = False,
    list_commands: bool = False,
):
    """
    Manipulate and inspect LensKit pipelines, data, and environments.
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

    if list_commands:
        _list_commands(ctx, lenskit)
        return

    if ctx.invoked_subcommand is None:
        _log.error("no command specified")
        print(lenskit.get_help(ctx))
        sys.exit(2)


@lenskit.command("version")
def version():
    """
    Print LensKit version info.
    """
    console.print(f"LensKit version [bold cyan]{__version__}[/bold cyan].")


def _list_commands(ctx: click.Context, cmd: click.Group | click.Command, prefix=""):
    name = prefix + cmd.name
    print(name)
    if isinstance(cmd, click.Group):
        for kid in cmd.list_commands(ctx):
            _list_commands(ctx, cmd.commands[kid], f"{name} ")


cli_plugins = entry_points(group="lenskit.cli.plugins")
for plugin in cli_plugins:
    try:
        cmd = plugin.load()
    except ImportError as e:
        warnings.warn(f"cannot load CLI {plugin.name}: {e}", ImportWarning)

    lenskit.add_command(cmd)
