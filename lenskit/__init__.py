"""
The LensKit package.
"""


from lenskit.algorithms import *  # noqa: F401,F403


class DataWarning(UserWarning):
    """
    Warning raised for detectable problems with input data.
    """
    pass
