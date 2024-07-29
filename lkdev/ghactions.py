# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Manage GitHub Actions workflows and template infrastructure

Usage:
    render-test-workflow.py [-v] [-o FILE] --render WORKFLOW

Options:
    -v, --verbose           verbose logging
    --render WORKFLOW       render the workflow WORKFLOW
    -o FILE, --output=FILE  render to FILE (- for stdout)
"""

# pyright: strict
from __future__ import annotations

import logging
import sys
from importlib import import_module
from textwrap import dedent
from typing import Any, NotRequired, TypedDict

import yaml
from docopt import docopt

_log = logging.getLogger("lkdev.ghactions")


class script:
    source: str

    def __init__(self, source: str):
        self.source = dedent(source).strip() + "\n"

    @staticmethod
    def presenter(dumper: yaml.Dumper, script: script):
        return dumper.represent_scalar("tag:yaml.org,2002:str", script.source, style="|")  # type: ignore

    @classmethod
    def command(cls, args: list[str]):
        return cls(" ".join(args))


yaml.add_representer(script, script.presenter)

GHStep = TypedDict(
    "GHStep",
    {
        "id": NotRequired[str],
        "name": NotRequired[str],
        "uses": NotRequired[str],
        "if": NotRequired[str],
        "run": NotRequired[str | script],
        "shell": NotRequired["str"],
        "with": NotRequired[dict[str, str | int | bool | script]],
        "env": NotRequired[dict[str, str | int]],
    },
)

GHJob = TypedDict(
    "GHJob",
    {
        "name": str,
        "runs-on": str,
        "if": NotRequired[str],
        "outputs": NotRequired[dict[str, str]],
        "timeout-minutes": NotRequired[int],
        "strategy": NotRequired[dict[str, Any]],
        "defaults": NotRequired[dict[str, Any]],
        "needs": NotRequired[list[str]],
        "steps": NotRequired[list[GHStep]],
    },
)


def main():
    options = docopt(__doc__)
    level = logging.DEBUG if options["--verbose"] else logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)

    render = options["--render"]
    if render:
        render_workflow(render, options)


def render_workflow(name: str, options: dict[str, Any]):
    mod_name = f"lkdev.workflows.{name}"
    _log.info("loading module %s", mod_name)
    mod = import_module(mod_name)

    workflow = mod.workflow()

    outfn = options["--output"]
    if outfn == "-":
        out = sys.stdout
    elif outfn:
        _log.info("saving to %s", outfn)
        out = open(outfn, "wt")
    else:
        outfn = f".github/workflows/{name}.yml"
        _log.info("saving to %s", outfn)
        out = open(outfn, "wt")

    try:
        yaml.dump(workflow, out, allow_unicode=True, sort_keys=False)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
