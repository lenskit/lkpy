from pathlib import Path

import click

from lenskit.data.dataset import Dataset
from lenskit.data.movielens import load_movielens
from lenskit.logging import get_logger

_log = get_logger(__name__)


@click.command("convert")
@click.option("--movielens", "format", flag_value="movielens", help="convert MovieLens data")
@click.argument("src", type=Path)
@click.argument("dst", type=Path)
def convert(format: str | None, src: Path, dst: Path):
    """
    Convert data into the LensKit native format.
    """

    match format:
        case None:
            _log.error("no data format specified")
            raise RuntimeError("no data format")
        case "movielens":
            data = convert_movielens(src)
        case _:
            raise ValueError(f"unknown data format {format}")

    log = _log.bind(dst=str(dst))
    log.info("saving data in native format")
    data.save(dst)


def convert_movielens(source: Path) -> Dataset:
    log = _log.bind(src=str(source))
    log.info("loading MovieLens data")
    return load_movielens(source)
