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

Options:
    -v, --verbose
        Ouput verbose logging.
    -q, --quiet
        Print version only, without other text.
"""

import logging
import sys

from docopt import docopt

from lenskit._version import lenskit_version

options = docopt(__doc__ or "")

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

# we only want to print the version
if options["--quiet"]:
    print(version)
else:
    print("LensKit version", version)
