"""
Toolkit for recommender systems research, teaching, and more.
"""


from lenskit.algorithms import *  # noqa: F401,F403

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("lenskit")
except PackageNotFoundError:
    # package is not installed
    pass


class DataWarning(UserWarning):
    """
    Warning raised for detectable problems with input data.
    """
    pass


class ConfigWarning(UserWarning):
    """
    Warning raised for detectable problems with algorithm configurations.
    """
    pass
