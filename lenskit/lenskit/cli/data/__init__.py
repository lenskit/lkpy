import click

from .convert import convert
from .describe import describe


@click.group
def data():
    """
    Data conversion and processing commands.
    """
    pass


data.add_command(convert)
data.add_command(describe)
