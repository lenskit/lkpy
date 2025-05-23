#!/usr/bin/env python3
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
    version-tool.py [-v | -vv] --run PROGRAM [ARGS...]

Options:
    -v, --verbose
        Ouput verbose logging.
    -q, --quiet
        Print version only, without other text.
    --github
        Write output to GitHub Actions output.
    --run
        Run a program with an updated version.
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import tomlkit
from docopt import docopt

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

elif options["--run"]:
    ppt_file = Path("pyproject.toml")
    saved = ppt_file.with_suffix(".toml.saved")
    shutil.copy(ppt_file, saved)
    with ppt_file.open("rt") as tf:
        meta = tomlkit.load(tf)

    proj = meta["project"]
    proj["dynamic"].remove("version")
    proj["version"] = str(version)

    try:
        with ppt_file.open("wt") as tf:
            _log.debug("writing updated pyproject.toml")
            tomlkit.dump(meta, tf)

        subprocess.check_call([options["PROGRAM"]] + options["ARGS"])
    finally:
        _log.info("restoring pyproject.toml")
        os.replace(saved, ppt_file)

else:
    # we only want to print the version
    if options["--quiet"]:
        print(version)
    else:
        print("LensKit version", version)
