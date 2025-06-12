# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path
from subprocess import check_call

import click

from lenskit.logging import Task, get_logger, stdout_console

_log = get_logger(__name__)
_gh_out: Path | None = None


@click.command("task")
@click.option(
    "-l",
    "--label",
    metavar="LABEL",
    default="cli-task",
    help="Human-readable task label",
)
@click.argument("args", nargs=-1)
def task(label: str, args: list[str]):
    """
    Run a task with LensKit task tracking.
    """
    console = stdout_console()

    log = _log.bind(label=label)

    with Task(label) as task:
        log.info("running command %s", " ".join(args))
        check_call(args)

    console.print(task)
