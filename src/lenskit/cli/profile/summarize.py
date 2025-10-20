# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

import click
import pandas as pd

from lenskit.logging import stdout_console


@click.command("summarize")
@click.argument("FILE", type=Path, required=True)
def summarize_profile(file: Path):
    """
    Summarize a pipeline profile.
    """

    console = stdout_console()
    prof = pd.read_csv(file)
    # convert to ms
    prof = prof / 1_000_000
    summary = prof.agg(["mean", "std", "median"]).T

    console.print("[bold]Stage timing (ms):[/bold]")
    console.print(summary)
