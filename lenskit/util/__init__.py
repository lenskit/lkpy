"""
Miscellaneous utility functions.
"""

import logging
from copy import deepcopy

from ..algorithms import Algorithm
from .files import delete_sometime, fspath
from .accum import Accumulator
from .timing import Stopwatch
from .data import read_df_detect, write_parquet, load_ml_ratings

_log = logging.getLogger(__name__)


def clone(algo):
    """
    Clone an algorithm, but not its fitted data.  This is like
    :py:func:`scikit.base.clone`, but may not work on arbitrary SciKit estimators.
    LensKit algorithms are compatible with SciKit clone, however, so feel free
    to use that if you need more general capabilities.

    This function is somewhat derived from the SciKit one.

    >>> from lenskit.algorithms.basic import Bias
    >>> orig = Bias()
    >>> copy = clone(orig)
    >>> copy is orig
    False
    >>> copy.damping == orig.damping
    True
    """
    _log.debug('cloning %s', algo)
    if isinstance(algo, Algorithm) or hasattr(algo, 'get_params'):
        params = algo.get_params(deep=False)

        sps = dict([(k, clone(v)) for (k, v) in params.items()])
        return algo.__class__(**sps)
    elif isinstance(algo, list) or isinstance(algo, tuple):
        return [clone(a) for a in algo]
    else:
        return deepcopy(algo)


class LastMemo:
    def __init__(self, func):
        self.function = func
        self.memory = None
        self.result = None

    def __call__(self, arg):
        if arg is not self.memory:
            self.result = self.function(arg)
            self.memory = arg

        return self.result
