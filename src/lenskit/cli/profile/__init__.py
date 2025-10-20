# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import click

from .summarize import summarize_profile


@click.group
def profile():
    """
    Summarize and manipulate profile data.
    """
    pass


profile.add_command(summarize_profile)
