from pathlib import Path
from urllib.request import urlopen

import click

from lenskit.logging import get_logger

_log = get_logger(__name__)
ML_LOC = "http://files.grouplens.org/datasets/movielens/"


@click.command("fetch")
@click.option("--movielens", "source", flag_value="movielens", help="fetch MovieLens data")
@click.option("-D", "--data-dir", "dest", type=Path, help="directory for downloaded data")
@click.argument("name")
def fetch(format: str | None, dest: Path | None, name: str):
    """
    Convert data into the LensKit native format.
    """

    if dest is None:
        dest = Path()

    match format:
        case None:
            _log.error("no data source specified")
            raise RuntimeError("no data source")
        case "movielens":
            fetch_movielens(name, dest)
        case _:
            raise ValueError(f"unknown data format {format}")


def fetch_movielens(name: str, base_dir: Path):
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

    _log.info("downloading MovieLens data set %s", name)
    with zipfile.open("wb") as zf:
        res = urlopen(zipurl)
        block = res.read(8 * 1024 * 1024)
        while len(block):
            _log.debug("received %d bytes", len(block))
            zf.write(block)
            block = res.read(8 * 1024 * 1024)
