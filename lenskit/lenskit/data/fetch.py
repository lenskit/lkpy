# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import argparse
import logging
import sys
from pathlib import Path
from urllib.request import urlopen

_log = logging.getLogger("lenskit.data.fetch")

ML_LOC = "http://files.grouplens.org/datasets/movielens/"


def fetch_ml(name: str, base_dir: Path):
    """
    Fetch a MovieLens dataset.  The followings names are recognized:

    . ml-100k
    . ml-1m
    . ml-10m
    . ml-20m
    . ml-25m
    . ml-latest
    . ml-latest-small

    Args:
        name:
            The name of the dataset.
        base_dir:
            The base directory into which data should be downloaded.
    """
    zipname = f"{name}.zip"
    zipfile = base_dir / zipname
    zipurl = ML_LOC + zipname

    if zipfile.exists():
        _log.info("%s already exists", zipfile)
        return

    _log.info("downloading data set %s", name)
    with zipfile.open("wb") as zf:
        res = urlopen(zipurl)
        block = res.read(8 * 1024 * 1024)
        while len(block):
            _log.debug("received %d bytes", len(block))
            zf.write(block)
            block = res.read(8 * 1024 * 1024)


def _fetch_main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="+", help="the name(s) of the dataset to fetch")
    parser.add_argument(
        "--data-dir", metavar="DIR", help="save extracted data to DIR", default="data"
    )
    args = parser.parse_args()

    dir = Path(args.data_dir)
    _log.info("extracting data to %s", dir)
    for name in args.name:
        _log.info("fetching data set %s", name)
        if name.startswith("ml-"):
            fetch_ml(name, dir)
        else:
            _log.error("unknown data set %s", name)
            raise ValueError("invalid data set")


if __name__ == "__main__":
    _fetch_main()
