import sys
from zipfile import ZipFile
from urllib.request import urlopen
import argparse
from pathlib import Path
import logging

_log = logging.getLogger("lenskit.datasets.fetch")

ML_LOC = "http://files.grouplens.org/datasets/movielens/"
ML_DATASETS = {
    "ml-100k": "ml-100k/u.data",
    "ml-1m": "ml-1m/ratings.dat",
    "ml-10m": "ml-10M100K/ratings.dat",
    "ml-20m": "ml-20m/ratings.csv",
    "ml-25m": "ml-25m/ratings.csv",
    "ml-latest": "ml-latest/ratings.csv",
    "ml-latest-small": "ml-latest-small/ratings.csv",
}


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
            The base directory into which data should be extracted.
    """
    zipname = f"{name}.zip"
    zipfile = base_dir / zipname
    zipurl = ML_LOC + zipname

    test_file = base_dir / ML_DATASETS[name]
    if test_file.exists():
        _log.info(test_file, "already exists")
        return

    _log.info("downloading data set %s", name)
    with zipfile.open("wb") as zf:
        res = urlopen(zipurl)
        block = res.read(8 * 1024 * 1024)
        while len(block):
            _log.debug("received %d bytes", len(block))
            zf.write(block)
            block = res.read(8 * 1024 * 1024)

    _log.info("unpacking data set")
    with ZipFile(zipfile, "r") as zf:
        zf.extractall(base_dir)


def _fetch_main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="the name of the dataset to fetch")
    parser.add_argument(
        "--data-dir", metavar="DIR", help="save extracted data to DIR", default="data"
    )
    args = parser.parse_args()

    name = args.name
    _log.info("fetching data set %s", name)
    dir = Path(args.data_dir)
    _log.info("extracting data to %s", dir)
    if name.startswith("ml-"):
        fetch_ml(name, dir)
    else:
        _log.error("unknown data set %s", name)
        raise ValueError("invalid data set")


if __name__ == "__main__":
    _fetch_main()
