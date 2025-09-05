#!/usr/bin/env python
# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Manipulate LensKit versions.

Usage:
    version-tool.py [-v | -vv] [-q]
    version-tool.py [-v | -vv] --github
    version-tool.py [-v | -vv] --update [DIR]

Options:
    -v, --verbose
        Ouput verbose logging.
    -q, --quiet
        Print version only, without other text.
    --github
        Write output to GitHub Actions output.
    --update
        Update version files.
"""

import logging
import os
import sys
from pathlib import Path

import tomlkit
from docopt import docopt

if prj_root := os.environ.get("MISE_PROJECT_ROOT", None):
    sys.path.insert(0, f"{prj_root}/src")

from lenskit._version import lenskit_version

_log = logging.getLogger("lenskit.version-tool")

options = docopt(__doc__ or "", options_first=True)

# set up logging
verbosity = options["--verbose"]
if verbosity > 1:
    level = logging.DEBUG
elif verbosity:
    level = logging.INFO
else:
    level = logging.WARNING
logging.basicConfig(level=level, stream=sys.stderr)

# get the version
version = lenskit_version()

if options["--github"]:
    gh_file = os.environ["GITHUB_OUTPUT"]
    with open(gh_file, "at") as ghf:
        print(f"version={version}", file=ghf)

elif options["--update"]:
    stage_dir = Path(options["DIR"] or ".")

    ppt_file = stage_dir / "pyproject.toml"
    with ppt_file.open("rt") as tf:
        meta = tomlkit.load(tf)

    proj = meta["project"]
    proj["dynamic"].remove("version")  # type: ignore
    proj["version"] = str(version)  # type: ignore

    _log.info("updating %s", str(ppt_file))
    with ppt_file.open("wt") as tf:
        _log.debug("writing updated pyproject.toml")
        tomlkit.dump(meta, tf)

else:
    # we only want to print the version
    if options["--quiet"]:
        print(version)
    else:
        print("LensKit version", version)
