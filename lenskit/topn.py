import logging

import numpy as np
import pandas as pd

from .metrics.topn import *

_log = logging.getLogger(__name__)


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
        item ID.  The metrics are defined in :mod:`lenskit.metrics.topn`; they are
        re-exported from :mod:`lenskit.topn` for convenience.

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
