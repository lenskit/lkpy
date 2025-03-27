# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import sys
from importlib.metadata import entry_points

import click
import numpy as np

from lenskit import __version__
from lenskit.logging import LoggingConfig, console, get_logger
from lenskit.logging.processors import error_was_logged

__all__ = ["lenskit", "main", "version"]
_log = get_logger(__name__)


def main():
    """
    Run the main LensKit CLI.  This just delegates to :fun:`lenskit`, but pretty-prints errors.
    """
    np.set_printoptions(threshold=20)
    try:
        ec = lenskit.main(standalone_mode=False)
    except Exception as e:
        _log.error("LensKit command failed", exc_info=e)

        sys.exit(3)

    if isinstance(ec, int):
        sys.exit(ec)


@click.group("lenskit")
@click.option("-v", "--verbose", "verbosity", count=True, help="enable verbose logging output")
def lenskit(verbosity: int):
    """
    Data and pipeline operations with LensKit.
    """

    # this code is run before any other command logic, so we can do global setup
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


cli_plugins = entry_points(group="lenskit.cli.plugins")
for plugin in cli_plugins:
    cmd = plugin.load()
    lenskit.add_command(cmd)
