"""
LensKit statistical computations.
"""

import warnings

import numpy as np
from numpy.typing import ArrayLike

from lenskit.diagnostics import DataWarning


def damped_mean(xs: ArrayLike, damping: float | None) -> float:
    xs = np.asarray(xs)
    if damping is not None and damping > 0:
        return xs.sum().item() / (np.sum(np.isfinite(xs)).item() + damping)
    else:
        return xs.mean().item()


def gini(xs: ArrayLike) -> float:
    """
    Compute the Gini coefficient of an array of values.

    This is inspired by `Olivia Guest's implementation`_ and based on the
    `StatsDirect reference`_.  It does *not* include Olivia's zero adjustment;
    the Gini coefficient is fine with some zeros, so long as the sum is strictly
    positive.

    .. _Olivia Guest's implementation: https://github.com/oliviaguest/gini
    .. _StatsDirect reference: https://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    Args:
        xs:
            An array of values representing the “resource” allocated to each
            instance.

    Returns:
        The Gini coefficient of the values in ``xs``.
    """
    xs = np.asarray(xs)
    if np.any(xs < 0):
        warnings.warn(
            "Gini coefficient is not defined for negative values", DataWarning, stacklevel=2
        )

    n = len(xs)
    # we don't actually need to sort — argsort will assign ranks to values
    xs = np.sort(xs)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    ranks *= 2
    ranks -= n + 1
    num = np.sum(xs * ranks)
    denom = n * np.sum(xs, dtype=np.float64)
    if denom <= 0:
        warnings.warn(
            "Gini coefficient is not defined for non-positive totals", DataWarning, stacklevel=2
        )
    return num / denom
