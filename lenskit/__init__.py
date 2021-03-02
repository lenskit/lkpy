"""
The LensKit package.
"""


from lenskit.algorithms import *  # noqa: F401,F403

__version__ = '0.12.1'


class DataWarning(UserWarning):
    """
    Warning raised for detectable problems with input data.
    """
    pass
