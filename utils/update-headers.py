#!/usr/bin/env python3
# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Update headers in source files.

Usage:
    update-headers.py [--check-only] [--error-on-change] [-y YEAR] [FILE...]

Options:
    -y YEAR, --year=YEAR
        Set the copyright year.
    --check-only
        Check headers, but don't update them.
    --error-on-change
        Exit with an error when headers are changed.
    FILE
        File to check or update.
"""

import re
import subprocess as sp
import sys
from datetime import date
from pathlib import Path

from docopt import docopt
from unbeheader.headers import SUPPORTED_FILE_TYPES, update_header
from unbeheader.typing import CommentSkeleton, SupportedFileType

SUPPORTED_FILE_TYPES["rs"] = SupportedFileType(
    re.compile(r"((^//|[\r\n]//).*)*"),
    CommentSkeleton("//", "//"),
)

options = docopt(__doc__ or "")
year = options["--year"]
files = options["FILE"]

if year is None:
    today = date.today()
    year = today.year

if files:
    files = [Path(p) for p in files]
else:
    out = sp.check_output(["git", "ls-files", "*.py", "*.rs"])

    files = [Path(p.strip()) for p in re.split(r"\r?\n", out.decode()) if p.strip()]

n = 0
print("scanning", len(files), "files")
for file in files:
    if update_header(file, year, check=options["--check-only"]):
        n += 1

print("updated", n, "files")
if options["--error-on-change"] and n > 0:
    sys.exit(5)
