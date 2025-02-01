# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

from lenskit.data.movielens import load_movielens
from lenskit.logging import LoggingConfig, get_logger

_log = get_logger("lenskit.data.convert")


def convert_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose logging")
    parser.add_argument("--movielens", action="store_true", help="convert MovieLens data")
    parser.add_argument("SRC", type=Path, help="the data set to convert")
    parser.add_argument("DST", type=Path, help="where to save the converted data")
    args = parser.parse_args()

    lc = LoggingConfig()
    if args.verbose:
        lc.set_verbose(True)
    lc.apply()

    log = _log.bind(src=str(args.SRC))
    if args.movielens:
        log.info("loading MovieLens data")
        data = load_movielens(args.SRC)
    else:
        log.error("no valid data type specfied")
        raise RuntimeError("no data type")

    log = log.bind(dst=str(args.DST))
    log.info("saving data")
    data.save(args.DST)


if __name__ == "__main__":
    convert_main()
