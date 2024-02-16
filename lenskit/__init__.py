"""
The LensKit toolkit for recommender systems research.
"""


from lenskit.algorithms import *  # noqa: F401,F403

__version__ = '0.14.4'


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
