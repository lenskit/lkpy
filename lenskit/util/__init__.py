"""
Miscellaneous utility functions.
"""

import logging
from copy import deepcopy

from ..algorithms import Algorithm
from .log import log_to_notebook, log_to_stderr  # noqa: F401
from .timing import Stopwatch  # noqa: F401
from .data import read_df_detect  # noqa: F401
from .random import rng, init_rng, derivable_rng  # noqa: F401
from .parallel import proc_count  # noqa: F401

try:
    import resource
except ImportError:
    resource = None

_log = logging.getLogger(__name__)

__all__ = [
    'log_to_stderr', 'log_to_notebook',
    'Stopwatch',
    'read_df_detect',
    'rng', 'init_rng', 'derivable_rng',
    'proc_count',
    'clone'
]


def clone(algo):
    """
    Clone an algorithm, but not its fitted data.  This is like
    :py:func:`scikit.base.clone`, but may not work on arbitrary SciKit estimators.
    LensKit algorithms are compatible with SciKit clone, however, so feel free
    to use that if you need more general capabilities.

    This function is somewhat derived from the SciKit one.

    >>> from lenskit.algorithms.bias import Bias
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
    def __init__(self, func, check_type='identity'):
        self.function = func
        self.check = check_type
        self.memory = None
        self.result = None

    def __call__(self, arg):
        if not self._arg_is_last(arg):
            self.result = self.function(arg)
            self.memory = arg

        return self.result

    def _arg_is_last(self, arg):
        if self.check == 'identity':
            return arg is self.memory
        elif self.check == 'equality':
            return arg == self.memory


def last_memo(func=None, check_type='identity'):
    if func is None:
        return lambda f: LastMemo(f, check_type)
    else:
        return LastMemo(func, check_type)


def no_progress(obj, **kwargs):
    return obj


def max_memory():
    "Get the maximum memory use for this process"
    if resource:
        res = resource.getrusage(resource.RUSAGE_SELF)
        return "%.1f MiB" % (res.ru_maxrss,)
    else:
        return 'unknown'


def cur_memory():
    "Get the current memory use for this process"
    if resource:
        res = resource.getrusage(resource.RUSAGE_SELF)
        return "%.1f MiB" % (res.ru_idrss,)
    else:
        return 'unknown'
