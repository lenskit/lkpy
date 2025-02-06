from pathlib import Path
from urllib.request import urlopen

import click
from humanize import naturalsize

from lenskit.logging import get_logger

_log = get_logger(__name__)
ML_LOC = "http://files.grouplens.org/datasets/movielens/"


@click.command("fetch")
@click.option("--movielens", "source", flag_value="movielens", help="fetch MovieLens data")
@click.option("--force", is_flag=True, help="overwrite existing file")
@click.option("-D", "--data-dir", "dest", type=Path, help="directory for downloaded data")
@click.argument("name", nargs=-1)
def fetch(source: str | None, dest: Path | None, name: list[str], force: bool):
    """
    Convert data into the LensKit native format.
    """

    if dest is None:
        dest = Path()

    match source:
        case None:
            _log.error("no data source specified")
            raise click.UsageError("no data source specified")
        case "movielens":
            for n in name:
                fetch_movielens(n, dest, force)
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
    base_dir.mkdir(exist_ok=True, parents=True)

    log.info("downloading MovieLens data set")
    with zipfile.open("wb") as zf:
        res = urlopen(zipurl)
        block = res.read(8 * 1024 * 1024)
        while len(block):
            _log.debug("received %d bytes", len(block))
            zf.write(block)
            block = res.read(8 * 1024 * 1024)

    log.info("downloaded %s", naturalsize(zipfile.stat().st_size, binary=True))
