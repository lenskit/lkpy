# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import click

from lenskit.logging import LoggingConfig, console

from .data import data

__all__ = ["lenskit", "main"]


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


lenskit.add_command(data)
