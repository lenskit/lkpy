import click

from .convert import convert
from .describe import describe
from .fetch import fetch


@click.group
def data():
    """
    Data conversion and processing commands.
    """
    pass


data.add_command(convert)
data.add_command(describe)
data.add_command(fetch)
