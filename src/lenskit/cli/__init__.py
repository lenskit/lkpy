# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from importlib.metadata import entry_points

import click

from lenskit import __version__
from lenskit.logging import LoggingConfig, console

__all__ = ["lenskit", "main", "version"]


def main():
    """
    Run the main LensKit CLI.  This just delegates to :fun:`lenskit`, but pretty-prints errors.
    """
    try:
        lenskit()
    except Exception as e:
        console.print(e)


@click.group("lenskit")
@click.option("-v", "--verbose", "verbosity", count=True, help="enable verbose logging output")
def lenskit(verbosity: int):
    """
    Data and pipeline operations with LensKit.
    """
    lc = LoggingConfig()
    if verbosity:
        lc.set_verbose(verbosity)
    lc.apply()


@lenskit.command("version")
def version():
    """
    Print LensKit version info.
    """
    console.print(f"LensKit version [bold cyan]{__version__}[/bold cyan].")


cli_plugins = entry_points(group="lenskit.cli-plugins")
for plugin in cli_plugins:
    cmd = plugin.load()
    lenskit.add_command(cmd)
