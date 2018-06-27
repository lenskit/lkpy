"""
Top-N evaluation metrics.
"""

import pandas as pd
import numpy as np

from .. import check


def precision(recs, relevant):
    """
    Compute the precision of a set of recommendations.

    Args:
        recs(array-like): a sequence of recommended items
        relevant(set-like): the set of relevant items

    Returns:
        double: the fraction of recommended items that are relevant
    """
    check.check_value(not isinstance(relevant, set),
                      "set type not supported for relevant set",
                      warn=True)
    if len(recs) == 0:
        return None

    recs = pd.Series(recs)
    ngood = recs.isin(relevant).sum()
    return ngood / len(recs)


def recall(recs, relevant):
    """
    Compute the recall of a set of recommendations.

    Args:
        recs(array-like): a sequence of recommended items
        relevant(set-like): the set of relevant items
    """
    check.check_value(not isinstance(relevant, set),
                      "set type not supported for relevant set",
                      warn=True)
    if len(relevant) == 0:
        return np.nan

    recs = pd.Series(recs)
    ngood = recs.isin(relevant).sum()
    return ngood / len(relevant)


def recip_rank(recs, relevant):
    """
    Compute the reciprocal rank of the first relevant item in a recommendation list.

    Args:
        recs(array-like): a sequence of recommended items
        relevant(set-like): the set of relevant items
    """
    check.check_value(not isinstance(relevant, set),
                      "set type not supported for relevant set",
                      warn=True)
    good = np.isin(recs, relevant)
    # nonzero returns a tuple, we have one dimension
    (nzp,) = good.nonzero()
    if len(nzp) == 0:
        return 0.0
    else:
        return 1.0 / (nzp[0] + 1)
