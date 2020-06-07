import logging
import itertools as it
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

    def __init__(self, group_cols=None, progress=None):
        self.group_cols = group_cols
        self.metrics = [(_length, 'nrecs', {})]
        self.progress = progress

    def add_metric(self, metric, *, name=None, **kwargs):
        """
        Add a metric to the analysis.

        A metric is a function of two arguments: the a single group of the recommendation
        frame, and the corresponding truth frame.  The truth frame will be indexed by
        item ID.  The recommendation frame will be in the order in the data.  Many metrics
        are defined in :mod:`lenskit.metrics.topn`; they are re-exported from
        :mod:`lenskit.topn` for convenience.

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
        _log.info('analyzing %d recommendations (%d truth rows)', len(recs), len(truth))

        rec_key, truth_key = self._df_keys(recs.columns, truth.columns)

        _log.info('collecting truth data')
        truth_frames = dict((k, df.set_index('item'))
                            for (k, df)
                            in self._iter_grouped(truth, truth_key))
        _log.debug('found truth for %d users', len(truth_frames))

        mnames = [mn for (mf, mn, margs) in self.metrics]

        def list_measure_gen():
            for rk, tk, df in self._iter_grouped(recs, rec_key, truth_key):
                g_truth = truth_frames[tk]
                results = tuple(mf(df, g_truth, **margs) for (mf, mn, margs) in self.metrics)
                yield rk + results

        timer = Stopwatch()
        _log.info('collecting metric results')
        res = pd.DataFrame.from_records(list_measure_gen(), columns=rec_key + mnames)
        res.set_index(rec_key, inplace=True)
        _log.info('measured %d lists in %s', len(res), timer)

        if include_missing:
            _log.info('filling in missing user info')
            ug_cols = [c for c in rec_key if c not in truth_key]
            tcount = truth.reset_index().groupby(truth_key)['item'].count()
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

    def _df_keys(self, r_cols, t_cols):
        "Identify rec list and truth list keys."
        gcols = self.group_cols
        if gcols is None:
            gcols = [c for c in r_cols if c not in self.DEFAULT_SKIP_COLS]

        truth_key = [c for c in gcols if c in t_cols]
        rec_key = [c for c in gcols if c not in t_cols] + truth_key
        _log.info('using rec key columns %s', rec_key)
        _log.info('using truth key columns %s', truth_key)
        return rec_key, truth_key

    def _iter_grouped(self, df, *keys):
        # try to do some memory optimization for groupby
        key = keys[0]
        kc_pos = dict((k, i) for (i, k) in enumerate(key))

        _log.debug('reindexing by %s', key)
        dfs = df.set_index(key, append=True)
        dfs = dfs.reorder_levels([i+1 for i in range(len(key))] + [0])
        assert dfs.index.names[:-1] == key
        dfs = dfs.sort_index()

        if self.progress is not None:
            _log.debug('counting')
            n = len(dfs.index.droplevel(-1).unique())
            prog = self.progress(total=n)
        else:
            prog = None

        _log.debug('iterating')
        for rk, df in dfs.groupby(level=key):
            if not isinstance(rk, tuple):
                rk = (rk,)
            row = [rk]
            for ok in keys[1:]:
                row.append(tuple(rk[kc_pos[kc]] for kc in ok))

            row.append(df.reset_index(key, drop=True))
            if prog is not None:
                prog.update()
            yield row

        if prog is not None:
            prog.close()
