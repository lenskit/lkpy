# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os
import sys
from importlib.metadata import entry_points
from pathlib import Path

import click
import numpy as np

from lenskit import __version__
from lenskit.logging import LoggingConfig, console, get_logger
from lenskit.random import init_global_rng, load_seed

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
@click.option(
    "--seed-file",
    type=Path,
    metavar="FILE",
    help="Load random seed from FILE (key: random.seed)",
    envvar="LK_SEED_FILE",
)
def lenskit(verbosity: int, seed_file: Path | None):
    """
    Data and pipeline operations with LensKit.
    """

    # this code is run before any other command logic, so we can do global setup
    lc = LoggingConfig()
    if verbosity:
        lc.set_verbose(verbosity)
    lc.apply()

    if seed_file is not None:  # pragma: nocover
        _log.info("loading RND seed from %s", seed_file)
        seed = load_seed(seed_file)
        init_global_rng(seed)
    elif seed := os.environ.get("LK_RANDOM_SEED", None):  # pragma: nocover
        _log.info("setting random seed from environment variable")
        init_global_rng(int(seed))


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
