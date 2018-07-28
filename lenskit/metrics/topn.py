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


def _dcg(scores, discount=np.log2):
    ranks = np.arange(1, len(scores) + 1)
    disc = np.maximum(discount(ranks), 1)
    return np.dot(scores, 1.0/disc)


def ndcg(scores, items=None, discount=np.log2):
    """
    Compute the Normalized Discounted Cumulative Gain of a series of scores.  These should be
    relevance scores; they can be :math:`{0,1}` for binary relevance data.

    Args:
        scores(pd.Series or array-like):
            relevance scores for items. If ``items`` is ``None``, these should be in order
            of recommendation; if ``items`` is not ``None``, then this must be a
            :py:class:`pandas.Series` indexed by item ID.
        items(array-like):
            the list of item IDs, if the item list and score list is to be provided separately.
        discount(ufunc):
            the rank discount function.  Each item's score will be divided the discount of its rank,
            if the discount is greater than 1.

    Returns:
        The nDCG of the scored items.
    """

    if not isinstance(scores, pd.Series):
        check.check_value(items is None, "scores must be Series when items provided")
        scores = pd.Series(scores)

    if items is None:
        actual = _dcg(scores, discount)
        iscores = scores.sort_values(ascending=False)
        ideal = _dcg(iscores, discount)
    else:
        ascores = scores.reindex(items, fill_value=0)
        actual = _dcg(ascores, discount)
        iscores = scores.sort_values(ascending=True)
        iscores = iscores[:len(items)]
        ideal = _dcg(iscores, discount)

    if ideal > 0:
        return actual / ideal
    else:
        return 0
