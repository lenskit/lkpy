"""
Miscellaneous utility functions.
"""


def compute(x):
    """
    Resolve a possibly-deferred value.  This is useful for e.g. forcing a Dask
    computation to be resolved, in code that should work transparently on both
    Pandas and Dask dataframes.

    :param x: the object to resolve
    :returns: the resolution of `x`; if `x` has a `compute()` method, that is
              called; otherwise `x` is returned.
    """
    if hasattr(x, 'compute'):
        return x.compute()
    else:
        return x
