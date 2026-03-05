# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import click

from .convert import convert
from .describe import describe
from .fetch import fetch
from .split import split
from .subset import subset


@click.group
def data():
    """
    Data conversion and processing commands.
    """
    pass


data.add_command(convert)
data.add_command(describe)
data.add_command(fetch)
data.add_command(subset)
data.add_command(split)
