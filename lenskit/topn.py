import logging
import warnings
from collections import OrderedDict as od

import numpy as np
import pandas as pd

from .metrics.topn import *
from .util import Stopwatch

_log = logging.getLogger(__name__)


def _length(df, *args, **kwargs):
    return float(len(df))


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

    .. warning::
       Currently, RecListAnalysis will silently drop users who received
       no recommendations.  We are working on an ergonomic API for fixing
       this problem.

    Args:
        group_cols(list):
            The columns to group by, or ``None`` to use the default.
    """

    DEFAULT_SKIP_COLS = ['item', 'rank', 'score', 'rating']

    def __init__(self, group_cols=None):
        self.group_cols = group_cols
        self.metrics = [(_length, 'nrecs', {})]

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

    def compute(self, recs, truth, *, include_missing=False):
        """
        Run the analysis.  Neither data frame should be meaningfully indexed.

        Args:
            recs(pandas.DataFrame):
                A data frame of recommendations.
            truth(pandas.DataFrame):
                A data frame of ground truth (test) data.
            include_missing(bool):
                ``True`` to include users from truth missing from recs.
                Matches are done via group columns that appear in both
                ``recs`` and ``truth``.

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

        ti_bcols = [c for c in gcols if c in truth.columns]
        ti_cols = ti_bcols + ['item']

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
            meta = dict((mn, 'f8') for (_f, mn, _a) in self.metrics)
            _log.debug('using meta %s', meta)
            res = grouped.apply(worker, meta=meta)
            res = res.compute()
        elif hasattr(grouped, 'progress_apply'):
            # Pandas has been patched with TQDM, use it
            res = grouped.progress_apply(worker)
        else:
            res = grouped.apply(worker)

        res.reset_index(level=-1, drop=True, inplace=True)
        _log.info('analyzed %d lists in %s', len(res), timer)
        if include_missing:
            _log.info('filling in missing user info')
            ug_cols = [c for c in gcols if c not in ti_bcols]
            tcount = truth.reset_index().groupby(ti_bcols)['item'].count()
            tcount.name = 'ntruth'
            _log.debug('res index levels: %s', res.index.names)
            if ug_cols:
                _log.debug('regrouping by %s to fill', ug_cols)
                res = res.groupby(level=ug_cols).apply(lambda f: f.reset_index(ug_cols, drop=True).join(tcount, how='outer'))
            else:
                _log.debug('no ungroup cols, directly merging to fill')
                res = res.join(tcount, how='outer')
            _log.debug('final columns: %s', res.columns)
            _log.debug('index levels: %s', res.index.names)
            res['ntruth'] = res['ntruth'].fillna(0)
            res['nrecs'] = res['nrecs'].fillna(0)

        return res
