import click

from .convert import convert


@click.group
def data():
    """
    Data conversion and processing commands.
    """
    pass


data.add_command(convert)
