# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# workaround for annoying coverage.py bug with path handling
import os
import sqlite3

if __name__ == "__main__":
    con = sqlite3.connect(".coverage")
    try:
        con.execute(r"UPDATE file SET path = replace(path, '\', '/')")
        con.commit()
    finally:
        con.close()

    os.rename(".coverage", "coverage.db")
