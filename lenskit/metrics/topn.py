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

    Returns:
        double: the fraction of relevant items that were recommended.
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
    This is used to compute MRR.

    Args:
        recs(array-like): a sequence of recommended items
        relevant(set-like): the set of relevant items

    Return:
        double: the reciprocal rank of the first relevant item.
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
    disc = discount(ranks)
    np.maximum(disc, 1, out=disc)
    np.reciprocal(disc, out=disc)
    return np.dot(scores, disc)


def dcg(scores, discount=np.log2):
    """
    Compute the Discounted Cumulative Gain of a series of recommended items with rating scores.
    These should be relevance scores; they can be :math:`{0,1}` for binary relevance data.

    Discounted cumultative gain is computed as:

    .. math::
        \\begin{align*}
        \\mathrm{DCG}(L,u) & = \\sum_{i=1}^{|L|} \\frac{r_{ui}}{d(i)}
        \\end{align*}

    You will usually want *normalized* discounted cumulative gain; this is

    .. math::
        \\begin{align*}
        \\mathrm{nDCG}(L, u) & = \\frac{\\mathrm{DCG}(L,u)}{\\mathrm{DCG}(L_{\\mathrm{ideal}}, u)}
        \\end{align*}

    Compute that by computing the DCG of the recommendations & the test data, then merge the results
    and divide.  The :py:func:`compute_ideal_dcgs` function is helpful for preparing that data.

    Args:
        scores(array-like):
            The utility scores of a list of recommendations, in recommendation order.
        discount(ufunc):
            the rank discount function.  Each item's score will be divided the discount of its rank,
            if the discount is greater than 1.

    Returns:
        double: the DCG of the scored items.
    """

    scores = np.nan_to_num(scores, copy=False)
    return _dcg(scores, discount)


def compute_ideal_dcgs(ratings, discount=np.log2):
    """
    Compute the ideal DCG for rating data.  This groups the rating data by everything *except* its
    ``item`` and ``rating`` columns, sorts each group by rating, and computes the DCG.

    Args:
        ratings(pandas.DataFrame):
            A rating data frame with ``item``, ``rating``, and other columns.

    Returns:
        pandas.DataFrame: The data frame of DCG values.  The ``item`` and ``rating`` columns in
            ``ratings`` are replaced by an ``ideal_dcg`` column.
    """

    def idcg(s):
        return dcg(s.sort_values(ascending=False), discount=discount)

    cols = [c for c in ratings.columns if c not in ('item', 'rating')]

    res = ratings.groupby(cols).rating.agg(idcg)
    return res.reset_index(name='ideal_dcg')
