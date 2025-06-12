# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

import click

from lenskit.config import load_configuration
from lenskit.logging import get_logger, stdout_console

_log = get_logger(__name__)
_gh_out: Path | None = None


@click.command("config")
@click.option(
    "-R",
    "--project-root",
    type=Path,
    metavar="DIR",
    help="Path to the project root to find lenskit.toml.",
)
def config(project_root: Path | None):
    """
    Inspect LensKit configuration.
    """
    console = stdout_console()
    config = load_configuration(cfg_dir=project_root)

    console.print_json(config.model_dump_json(indent=2))
