from io import StringIO
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown

from lenskit.data.dataset import Dataset
from lenskit.data.movielens import load_movielens
from lenskit.data.summary import save_stats
from lenskit.logging import get_logger

_log = get_logger(__name__)


@click.command("describe")
@click.option("--movielens", "format", flag_value="movielens", help="describe MovieLens data")
@click.argument("path", type=Path)
def describe(format: str | None, path: Path):
    """
    Describe a data set.
    """

    log = _log.bind(path=str(path))

    match format:
        case None:
            log.info("loading LensKit native data")
            data = Dataset.load(path)
        case "movielens":
            log.info("loading MovieLens data")
            data = load_movielens(path)
        case _:
            raise ValueError(f"unknown data format {format}")

    console = Console()
    out = StringIO()
    save_stats(data, out)
    console.print(Markdown(out.getvalue()), width=min(console.width, 80))


def convert_movielens(source: Path) -> Dataset:
    log = _log.bind(src=str(source))
    log.info("loading MovieLens data")
    return load_movielens(source)
