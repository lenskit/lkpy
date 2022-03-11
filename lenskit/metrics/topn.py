"""
Top-N evaluation metrics.
"""

import logging
import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def bulk_impl(metric):
    """
    Decorator to register a bulk implementation for a metric.
    """
    def wrap(impl):
        metric.bulk_score = impl
        return impl

    return wrap


def precision(recs, truth, k=None):
    """
    Compute recommendation precision.  This is computed as:

    .. math::
        \\frac{|L \\cap I_u^{\\mathrm{test}}|}{|L|}

    In the uncommon case that ``k`` is specified and ``len(recs) < k``, this metric uses
    ``len(recs)`` as the denominator.

    This metric has a bulk implementation.
    """
    if k is not None:
        recs = recs.iloc[:k]

    nrecs = len(recs)
    if nrecs == 0:
        return None

    ngood = recs['item'].isin(truth.index).sum()
    return ngood / nrecs


@bulk_impl(precision)
def _bulk_precision(recs, truth, k=None):
    if k is not None:
        recs = recs[recs['rank'] <= k]
        lcounts = pd.Series(k, index=recs['LKRecID'].unique())
        lcounts.index.name = 'LKRecID'
    else:
        lcounts = recs.groupby(['LKRecID'])['item'].count()

    good = recs.join(truth, on=['LKTruthID', 'item'], how='inner')
    gcounts = good.groupby(['LKRecID'])['item'].count()

    lcounts, gcounts = lcounts.align(gcounts, join='left', fill_value=0)

    return gcounts / lcounts


def recall(recs, truth, k=None):
    """
    Compute recommendation recall.  This is computed as:

    .. math::
        \\frac{|L \\cap I_u^{\\mathrm{test}}|}{\\operatorname{max}\\{|I_u^{\\mathrm{test}}|, k\\}}

    This metric has a bulk implementation.
    """
    nrel = len(truth)
    if nrel == 0:
        return None

    if k is not None:
        nrel = min(nrel, k)
        recs = recs.iloc[:k]

    ngood = recs['item'].isin(truth.index).sum()
    return ngood / nrel


@bulk_impl(recall)
def _bulk_recall(recs, truth, k=None):
    tcounts = truth.reset_index().groupby('LKTruthID')['item'].count()

    if k is not None:
        _log.debug('truncating to k for recall')
        tcounts = np.minimum(tcounts, k)
        recs = recs[recs['rank'] <= k]

    good = recs.join(truth, on=['LKTruthID', 'item'], how='inner')
    gcounts = good.groupby('LKRecID')['item'].count()

    # we need all lists, because some might have no truth (oops), some no recs (also oops)
    lists = recs[['LKRecID', 'LKTruthID']].drop_duplicates()

    scores = lists.join(gcounts.to_frame('ngood'), on='LKRecID', how='left')
    scores['ngood'].fillna(0, inplace=True)
    scores = scores.join(tcounts.to_frame('nrel'), on='LKTruthID', how='left')
    scores = scores.set_index('LKRecID')
    return scores['ngood'] / scores['nrel']


def hit(recs, truth, k=None):
    """
    Compute whether or not a list is a hit; any list with at least one relevant item in the
    first :math:`k` positions (:math:`L_{\\le k} \\cap I_u^{\\mathrm{test}} \\ne \\emptyset`)
    is scored as 1, and lists with no relevant items as 0.  When averaged over the recommendation
    lists, this computes the *hit rate* :cite:p:`Deshpande2004-ht`.

    .. math::
        \\frac{|L \\cap I_u^{\\mathrm{test}}|}{\\operatorname{max}\\{|I_u^{\\mathrm{test}}|, k\\}}

    This metric has a bulk implementation.
    """
    nrel = len(truth)
    if nrel == 0:
        return None

    if k is not None:
        nrel = min(nrel, k)
        recs = recs.iloc[:k]

    good = recs['item'].isin(truth.index)
    if np.any(good):
        return 1
    else:
        return 0


@bulk_impl(hit)
def _bulk_hit(recs, truth, k=None):
    tcounts = truth.reset_index().groupby('LKTruthID')['item'].count()

    if k is not None:
        _log.debug('truncating to k for recall')
        tcounts = np.minimum(tcounts, k)
        recs = recs[recs['rank'] <= k]

    good = recs.join(truth, on=['LKTruthID', 'item'], how='inner')
    gcounts = good.groupby('LKRecID')['item'].count()

    # we need all lists, because some might have no truth (oops), some no recs (also oops)
    lists = recs[['LKRecID', 'LKTruthID']].drop_duplicates()

    scores = lists.join(gcounts.to_frame('ngood'), on='LKRecID', how='left')
    scores['ngood'].fillna(0, inplace=True)

    scores = scores.join(tcounts.to_frame('nrel'), on='LKTruthID', how='left')
    scores = scores.set_index('LKRecID')

    good = scores['ngood'] > 0
    good = good.astype('f4')
    good[scores['nrel'] == 0] = np.nan
    return good


def recip_rank(recs, truth, k=None):
    """
    Compute the reciprocal rank :cite:p:`Kantor1997-lm` of the first relevant
    item in a list of recommendations. Let :math:`\\kappa` denote the 1-based
    rank of the first relevant item in :math:`L`, with :math:`\\kappa=\\infty`
    if none of the first :math:`k` items in :math:`L` are relevant; then the
    reciprocal rank is :math:`1 / \\kappa`. If no elements are relevant, the
    reciprocal rank is therefore 0.  :cite:t:`Deshpande2004-ht` call this the
    “reciprocal hit rate”.

    This metric has a bulk equivalent.
    """
    if k is not None:
        recs = recs.iloc[:k]

    good = recs['item'].isin(truth.index)
    npz, = np.nonzero(good.to_numpy())
    if len(npz):
        return 1.0 / (npz[0] + 1.0)
    else:
        return 0.0


@bulk_impl(recip_rank)
def _bulk_rr(recs, truth, k=None):
    # find everything with truth
    if k is not None:
        recs = recs[recs['rank'] <= k]
    joined = recs.join(truth, on=['LKTruthID', 'item'], how='inner')
    # compute min ranks
    ranks = joined.groupby('LKRecID')['rank'].agg('min')
    # reciprocal ranks
    scores = 1.0 / ranks
    _log.debug('have %d scores with MRR %.3f', len(scores), scores.mean())
    # fill with zeros
    rec_ids = recs['LKRecID'].unique()
    scores = scores.reindex(rec_ids, fill_value=0.0)
    _log.debug('filled to get %s scores w/ MRR %.3f', len(scores), scores.mean())
    # and we're done
    return scores


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
    scores = np.nan_to_num(scores)
    ranks = np.arange(1, len(scores) + 1)
    disc = discount(ranks)
    np.maximum(disc, 1, out=disc)
    np.reciprocal(disc, out=disc)
    return np.dot(scores, disc)


def _fixed_dcg(n, discount=np.log2):
    ranks = np.arange(1, n+1)
    disc = discount(ranks)
    disc = np.maximum(disc, 1)
    disc = np.reciprocal(disc)
    return np.sum(disc)


def dcg(recs, truth, discount=np.log2):
    """
    Compute the **unnormalized** discounted cumulative gain :cite:p:`Jarvelin2002-xf`.

    Discounted cumultative gain is computed as:

    .. math::
        \\begin{align*}
        \\mathrm{DCG}(L,u) & = \\sum_{i=1}^{|L|} \\frac{r_{ui}}{d(i)}
        \\end{align*}

    Unrated items are assumed to have a utility of 0; if no rating values are provided in the
    truth frame, item ratings are assumed to be 1.

    Args:
        recs: The recommendation list.
        truth: The user's test data.
        discount(ufunc):
            The rank discount function.  Each item's score will be divided the discount of its rank,
            if the discount is greater than 1.
    """

    tpos = truth.index.get_indexer(recs['item'])
    tgood = tpos >= 0
    if 'rating' in truth.columns:
        # make an array of ratings for this rec list
        r_rates = truth['rating'].values[tpos]
        r_rates[tpos < 0] = 0
        achieved = _dcg(r_rates, discount)
    else:
        achieved = _dcg(tgood, discount)

    return achieved


def ndcg(recs, truth, discount=np.log2, k=None):
    """
    Compute the normalized discounted cumulative gain :cite:p:`Jarvelin2002-xf`.

    Discounted cumultative gain is computed as:

    .. math::
        \\begin{align*}
        \\mathrm{DCG}(L,u) & = \\sum_{i=1}^{|L|} \\frac{r_{ui}}{d(i)}
        \\end{align*}

    Unrated items are assumed to have a utility of 0; if no rating values are provided in the
    truth frame, item ratings are assumed to be 1.

    This is then normalized as follows:

    .. math::
        \\begin{align*}
        \\mathrm{nDCG}(L, u) & = \\frac{\\mathrm{DCG}(L,u)}{\\mathrm{DCG}(L_{\\mathrm{ideal}}, u)}
        \\end{align*}

    This metric has a bulk implementation.

    Args:
        recs: The recommendation list.
        truth: The user's test data.
        discount(numpy.ufunc):
            The rank discount function.  Each item's score will be divided the discount of its rank,
            if the discount is greater than 1.
        k(int):
            The maximum list length.
    """

    if k is not None:
        recs = recs.iloc[:k]

    tpos = truth.index.get_indexer(recs['item'])

    if 'rating' in truth.columns:
        i_rates = np.sort(truth.rating.values)[::-1]
        if k is not None:
            i_rates = i_rates[:k]
        ideal = _dcg(i_rates, discount)
        # make an array of ratings for this rec list
        r_rates = truth['rating'].values[tpos]
        r_rates[tpos < 0] = 0
        achieved = _dcg(r_rates, discount)
    else:
        ntrue = len(truth)
        if k is not None and ntrue > k:
            ntrue = k
        ideal = _fixed_dcg(ntrue, discount)
        tgood = tpos >= 0
        achieved = _dcg(tgood, discount)

    return achieved / ideal


@bulk_impl(ndcg)
def _bulk_ndcg(recs, truth, discount=np.log2, k=None):
    if 'rating' not in truth.columns:
        truth = truth.assign(rating=np.ones(len(truth), dtype=np.float32))

    ideal = truth.groupby(level='LKTruthID')['rating'].rank(method='first', ascending=False)
    if k is not None:
        ideal = ideal[ideal <= k]
    ideal = discount(ideal)
    ideal = np.maximum(ideal, 1)
    ideal = truth['rating'] / ideal
    ideal = ideal.groupby(level='LKTruthID').sum()
    ideal.name = 'ideal'

    list_ideal = recs[['LKRecID', 'LKTruthID']].drop_duplicates()
    list_ideal = list_ideal.join(ideal, on='LKTruthID', how='left')
    list_ideal = list_ideal.set_index('LKRecID')

    if k is not None:
        recs = recs[recs['rank'] <= k]
    rated = recs.join(truth, on=['LKTruthID', 'item'], how='inner')
    rd = discount(rated['rank'])
    rd = np.maximum(rd, 1)
    rd = rated['rating'] / rd
    rd = rated[['LKRecID']].assign(util=rd)
    dcg = rd.groupby(['LKRecID'])['util'].sum().reset_index(name='dcg')
    dcg = dcg.set_index('LKRecID')

    dcg = dcg.join(list_ideal, how='outer')
    dcg['ndcg'] = dcg['dcg'].fillna(0) / dcg['ideal']

    return dcg['ndcg']
