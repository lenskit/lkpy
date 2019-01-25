import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


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


class RecListAnalysis:
    """
    Compute one or more top-N metrics over recommendation lists.

    This method groups the recommendations by the specified columns,
    and computes the metric over each group.  The default set of grouping
    columns is all columns *except* the following:

    * ``item``
    * ``rank``
    * ``score``
    * ``rating``

    The truth frame, ``truth``, is expected to match over (a subset of) the
    grouping columns, and contain at least an ``item`` column.  If it also
    contains a ``rating`` column, that is used as the users' rating for
    metrics that require it; otherwise, a rating value of 1 is assumed.

    Args:
        group_cols(list):
            The columns to group by, or ``None`` to use the default.
    """

    DEFAULT_SKIP_COLS = ['item', 'rank', 'score', 'rating']

    def __init__(self, group_cols=None):
        self.group_cols = group_cols
        self.metrics = []

    def add_metric(self, metric, *, name=None, **kwargs):
        """
        Add a metric to the analysis.

        A metric is a function of two arguments: the a single group of the recommendation
        frame, and the corresponding truth frame.  The truth frame will be indexed by
        item ID.  The metric functions in this module are usable; the
        :mod:`lenskit.metrics.topn` module provides underlying implementations
        for some of them that operate on more raw structures.

        Args:
            metric: The metric to compute.
            name: The name to assign the metric. If not provided, the function name is used.
            **kwargs: Additional arguments to pass to the metric.
        """
        if name is None:
            name = metric.__name__

        self.metrics.append((metric, name, kwargs))

    def compute(self, recs, truth):
        """
        Run the analysis.  Neither data frame should be meaningfully indexed.

        Args:
            recs(pandas.DataFrame):
                A data frame of recommendations.
            truth(pandas.DataFrame):
                A data frame of ground truth (test) data.

        Returns:
            pandas.DataFrame: The results of the analysis.
        """
        gcols = self.group_cols
        if gcols is None:
            gcols = [c for c in recs.columns if c not in self.DEFAULT_SKIP_COLS]

        ti_cols = [c for c in gcols if c in truth.columns]
        truth = truth.set_index(ti_cols)

        _log.info('analyzing %d recommendations (%d truth rows)', len(recs), len(truth))
        _log.info('using group columns %s', gcols)
        _log.info('using truth ID columns %s', ti_cols)

        return recs.groupby(gcols).apply(self._group_compute, truth=truth, cols=ti_cols)

    def _group_compute(self, recs, truth, cols):
        key = recs.loc[:, cols]
        key = key.iloc[[0], :]
        g_truth = key.join(truth, on=cols)
        g_truth = g_truth.set_index('item')
        vals = dict((k, f(recs, g_truth, **args)) for (f, k, args) in self.metrics)
        return pd.Series(vals)


class UnratedCandidates:
    """
    Candidate selector that selects unrated items from a training set.

    Args:
        training(pandas.DataFrame):
            the training data; must have ``user`` and ``item`` columns.
    """

    def __init__(self, training):
        self.training = training.set_index('user').item
        self.items = training.item.unique()

    def __call__(self, user, *args, **kwargs):
        urates = self.training.loc[user]
        return np.setdiff1d(self.items, urates)
