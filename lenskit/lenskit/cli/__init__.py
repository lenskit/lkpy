import click

from lenskit.logging import LoggingConfig

from .data import data

__all__ = ["lenskit"]


@click.group("lenskit")
@click.option("-v", "--verbose", "verbosity", count=True, help="enable verbose logging output")
def lenskit(verbosity: int):
    """
    Data and pipeline operations with LensKit.
    """
    lc = LoggingConfig()
    if verbosity:
        lc.set_verbose(verbosity)
    lc.apply()


lenskit.add_command(data)
