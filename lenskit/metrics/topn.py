"""
Top-N evaluation metrics.
"""

import pandas as pd
import numpy as np


def precision(recs, truth):
    """
    Compute the precision of a set of recommendations.
    """

    nrecs = len(recs)
    if nrecs == 0:
        return None

    ngood = recs['item'].isin(truth.index).sum()
    return ngood / nrecs


def recall(recs, truth):
    """
    Compute the recall of a set of recommendations.
    """

    nrel = len(truth)
    if nrel == 0:
        return None

    ngood = recs['item'].isin(truth.index).sum()
    return ngood / nrel


def recip_rank(recs, truth):
    """
    Compute the reciprocal rank of the first relevant item in a list of recommendations.

    If no elements are relevant, the reciprocal rank is 0.
    """
    good = recs['item'].isin(truth.index)
    npz, = np.nonzero(good)
    if len(npz):
        return 1.0 / (npz[0] + 1.0)
    else:
        return 0.0


def _dcg(scores, discount=np.log2):
    """
    Compute the Discounted Cumulative Gain of a series of recommended items with rating scores.
    These should be relevance scores; they can be :math:`{0,1}` for binary relevance data.

    This is not a true top-N metric, but is a utility function for other metrics.

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
    ranks = np.arange(1, len(scores) + 1)
    disc = discount(ranks)
    np.maximum(disc, 1, out=disc)
    np.reciprocal(disc, out=disc)
    return np.dot(scores, disc)


def ndcg(recs, truth, discount=np.log2):
    """
    Compute the normalized discounted cumulative gain.

    Discounted cumultative gain is computed as:

    .. math::
        \\begin{align*}
        \\mathrm{DCG}(L,u) & = \\sum_{i=1}^{|L|} \\frac{r_{ui}}{d(i)}
        \\end{align*}

    This is then normalized as follows:

    .. math::
        \\begin{align*}
        \\mathrm{nDCG}(L, u) & = \\frac{\\mathrm{DCG}(L,u)}{\\mathrm{DCG}(L_{\\mathrm{ideal}}, u)}
        \\end{align*}

    Args:
        recs: The recommendation list.
        truth: The user's test data.
        discount(ufunc):
            The rank discount function.  Each item's score will be divided the discount of its rank,
            if the discount is greater than 1.
    """
    if 'rating' in truth.columns:
        ideal = _dcg(truth.rating.sort_values(ascending=False), discount)
        merged = recs[['item']].join(truth[['rating']], on='item', how='left')
        achieved = _dcg(merged.rating, discount)
    else:
        ideal = _dcg(np.ones(len(truth)), discount)
        achieved = _dcg(recs.item.isin(truth.index), discount)

    return achieved / ideal
