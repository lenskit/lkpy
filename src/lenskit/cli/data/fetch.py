# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os
from contextlib import contextmanager
from pathlib import Path
from shutil import copyfileobj
from urllib.parse import urlparse
from urllib.request import urlopen

import click
from humanize import naturalsize

from lenskit.logging import get_logger

_log = get_logger(__name__)
ML_LOC = "http://files.grouplens.org/datasets/movielens/"
AZ_LOC = {
    "2023": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/{}.csv.gz",
    "2023-5core": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/{}.csv.gz",
    "2018": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/{}.csv",
    "2014": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_{}.csv",
}

MSW_BASE = "https://kdd.ics.uci.edu/databases/msweb/"


@click.command("fetch")
@click.option("--movielens", "source", flag_value="movielens", help="Fetch MovieLens data.")
@click.option("--amazon", "source", flag_value="amazon", help="Fetch Amazon ratings data.")
@click.option("--ms-web", "source", flag_value="ms-web", help="Fetch MS Web visit logs.")
@click.option("--edition", default="2023", help="Amazon ratings edition.")
@click.option("--core", is_flag=True, help="Fetch core instead of full data.")
@click.option("--force", is_flag=True, help="overwrite existing file")
@click.option(
    "-D", "--data-dir", "dest", default="data", type=Path, help="directory for downloaded data"
)
@click.argument("name", nargs=-1)
def fetch(source: str | None, dest: Path, name: list[str], force: bool, edition: str, core: bool):
    """
    Convert data into the LensKit native format.
    """

    if dest is None:
        dest = Path()

    dest.mkdir(exist_ok=True, parents=True)

    match source:
        case None:
            _log.error("no data source specified")
            raise click.UsageError("no data source specified")
        case "movielens":
            for n in name:
                fetch_movielens(n, dest, force)
        case "amazon":
            for n in name:
                fetch_amazon_ratings(edition, core, n, dest, force)
        case "ms-web":
            fetch_ms_web(dest, force)
        case _:
            raise ValueError(f"unknown data format {source}")


def fetch_movielens(name: str, base_dir: Path, force: bool):
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

    log = _log.bind(source="movielens", name=name, dest=str(zipfile))

    if zipfile.exists():
        if force:
            log.warning("output file already exists, ovewriting")
        else:
            log.info("output file already exists")
            return

    log.debug("ensuring parent directory exists")

    log.info("downloading MovieLens data set")
    with output_file(zipfile) as zf:
        res = urlopen(zipurl)
        copyfileobj(res, zf, 8 * 1024 * 1024)

    log.info("downloaded %s", naturalsize(zipfile.stat().st_size, binary=True))


def fetch_amazon_ratings(edition: str, core: bool, name: str, base_dir: Path, force: bool):
    key = edition + "-5core" if core else edition
    log = _log.bind(source="amazon", name=name, edition=edition, core=core)
    pat = AZ_LOC[key]

    url = pat.format(name)
    fn = os.path.basename(urlparse(url).path)

    out_path = base_dir / fn
    if out_path.exists() and not force:
        log.info("output file already exists")
        return

    log.info("downloading %s", fn)
    with output_file(out_path) as outf:
        res = urlopen(url)
        copyfileobj(res, outf, 8 * 1024 * 1024)

    log.info("downloaded %s", naturalsize(out_path.stat().st_size, binary=True))


def fetch_ms_web(base_dir: Path, force: bool):
    """
    Fetch an MS Web data set.
    """
    data_name = "anonymous-msweb.data.gz"
    test_name = "anonymous-msweb.test.gz"

    data_out = base_dir / data_name
    test_out = base_dir / test_name

    _log.info("fetching Anonymous MS Web dataset into %s", base_dir)

    if force or not data_out.exists():
        _log.info("fetching training set")
        with output_file(data_out) as outf:
            res = urlopen(MSW_BASE + data_name)
            copyfileobj(res, outf, 8 * 1024 * 1024)

    if force or not test_out.exists():
        _log.info("fetching test set")
        with output_file(test_out) as outf:
            res = urlopen(MSW_BASE + test_name)
            copyfileobj(res, outf, 8 * 1024 * 1024)


@contextmanager
def output_file(path: Path):
    """
    Open an output file and write.
    """
    pid = os.getpid()
    tmp = path.with_name(f".{path.name}.{pid}.tmp")
    _log.debug("opening temp file", name=tmp.name)
    try:
        with open(tmp, "wb") as file:
            yield file
        _log.debug("replacing real file", name=path.name)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            _log.debug("write failed, deleting temp file", name=tmp.name)
            tmp.unlink()
