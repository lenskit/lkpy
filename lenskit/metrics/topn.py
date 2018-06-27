"""
Top-N evaluation metrics.
"""


import pandas as pd
import numpy as np


def precision(recs, relevant):
    """
    Compute the precision of a set of recommendations.

    Args:
        recs(array-like): a sequence of recommended items
        relevant(set-like): the set of relevant items

    Returns:
        double: the fraction of recommended items that are relevant
    """
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
    pass


def avg_precision(recs, relevant):
    """
    Compute the average precision of a list of recommendations.

    Args:
        recs(array-like): a sequence of recommended items
        relevant(set-like): the set of relevant items
    """
    pass
