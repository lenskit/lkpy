import logging
import warnings
from collections import OrderedDict as od

import numpy as np
import pandas as pd

from .metrics.topn import *
from .util import Stopwatch

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
        item ID.  Many metrics are defined in :mod:`lenskit.metrics.topn`; they are
        re-exported from :mod:`lenskit.topn` for convenience.

        Args:
            metric: The metric to compute.
            name: The name to assign the metric. If not provided, the function name is used.
            **kwargs: Additional arguments to pass to the metric.
        """
        if name is None:
            name = metric.__name__

        self.metrics.append((metric, name, kwargs))

    def compute(self, recs, truth, *, progress=lambda x: x):
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
        using_dask = type(recs).__module__.startswith('dask.')

        _log.info('analyzing %d recommendations (%d truth rows)', len(recs), len(truth))
        gcols = self.group_cols
        if gcols is None:
            gcols = [c for c in recs.columns if c not in self.DEFAULT_SKIP_COLS]
        _log.info('using group columns %s', gcols)
        _log.info('ungrouped columns: %s', [c for c in recs.columns if c not in gcols])
        gc_map = dict((c, i) for (i, c) in enumerate(gcols))

        ti_cols = [c for c in gcols if c in truth.columns]
        ti_cols.append('item')

        _log.info('using truth ID columns %s', ti_cols)
        truth = truth.set_index(ti_cols)
        if not truth.index.is_unique:
            warnings.warn('truth frame does not have unique values')
        truth.sort_index(inplace=True)

        def worker(group):
            row_key = group.name
            if len(ti_cols) == len(gcols) + 1:
                tr_key = row_key
            else:
                tr_key = tuple([row_key[gc_map[c]] for c in ti_cols[:-1]])

            g_truth = truth.loc[tr_key, :]

            group_results = {}
            for mf, mn, margs in self.metrics:
                group_results[mn] = mf(group, g_truth, **margs)
            return pd.DataFrame(group_results, index=[0])

        timer = Stopwatch()
        grouped = recs.groupby(gcols)
        _log.info('computing analysis for %s lists',
                  len(grouped) if hasattr(grouped, '__len__') else 'many')

        if using_dask:
            # Dask group-apply requires metadata
            meta = dict((mn, 'f8') for (mf, mn, margs) in self.metrics)
            res = grouped.apply(worker, meta=meta)
            res = res.compute()
        else:
            res = grouped.apply(worker)

        res.reset_index(level=-1, drop=True, inplace=True)
        _log.info('analyzed %d lists in %s', len(res), timer)

        return res


class UnratedCandidates:
    """
    Candidate selector that selects unrated items from a training set.

    Args:
        training(pandas.DataFrame):
            the training data; must have ``user`` and ``item`` columns.
    """

    def __init__(self, training):
        warnings.warn('UnratedCandidates deprecated, use default item selector', DeprecationWarning)
        self.training = training.set_index('user').item
        self.items = training.item.unique()

    def __call__(self, user, *args, **kwargs):
        urates = self.training.loc[user]
        return np.setdiff1d(self.items, urates)
