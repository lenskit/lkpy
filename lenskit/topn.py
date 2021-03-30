import logging

import numpy as np
import pandas as pd

from .metrics.topn import *
from .util import Stopwatch

_log = logging.getLogger(__name__)


def _length(df, *args, **kwargs):
    return float(len(df))


class _RLAJob:
    def __init__(self, recs, truth, metrics):
        self.recs = recs
        self.truth = truth
        self.metrics = metrics

    def prepare(self, group_cols):
        rec_key, truth_key = _df_keys(self.recs.columns, self.truth.columns, group_cols)
        _log.info('numbering truth lists')
        truth_df = self.truth[truth_key].drop_duplicates()
        truth_df['LKTruthID'] = np.arange(len(truth_df))
        truth = pd.merge(truth_df, self.truth, on=truth_key).drop(columns=truth_key)
        _log.debug('truth lists:\n%s', truth_df)

        _log.info('numbering rec lists')
        rec_df = self.recs[rec_key].drop_duplicates()
        rec_df['LKRecID'] = np.arange(len(rec_df))
        rec_df = pd.merge(rec_df, truth_df, on=truth_key, how='left')
        recs = pd.merge(rec_df, self.recs, on=rec_key).drop(columns=rec_key)
        _log.debug('rec lists:\n%s', rec_df)

        _log.info('collecting truth data')
        truth = truth.set_index(['LKTruthID', 'item'])
        if not truth.index.is_unique:
            _log.warn('truth index not unique: may have duplicate items\n%s', truth)

        _log.debug('found truth for %d users', len(truth_df))
        self.truth = truth
        self.recs = recs
        self.rec_attrs = rec_df
        self.rec_key = rec_key
        self.truth_key = truth_key

    def compute(self, n_jobs=None):
        def worker(rdf):
            rk, tk = rdf.name
            tdf = self.truth.loc[tk]
            res = pd.Series(dict((mn, mf(rdf, tdf, **margs)) for (mf, mn, margs) in self.metrics))
            return res

        _log.debug('applying metrics')
        groups = self.recs.groupby(['LKRecID', 'LKTruthID'])
        if hasattr(groups, 'progress_apply'):
            result = groups.progress_apply(worker)
        else:
            result = groups.apply(worker)
        _log.debug('result frame:\n%s', result)
        _log.debug('transforming results')
        result = result.reset_index('LKTruthID', drop=True)
        result = self.rec_attrs.join(result, on='LKRecID').drop(columns=['LKRecID', 'LKTruthID'])
        _log.debug('result frame:\n%s', result)
        return result


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

    def __init__(self, group_cols=None, n_jobs=None):
        self.group_cols = group_cols
        self.metrics = [(_length, 'nrecs', {})]
        self.n_jobs = n_jobs

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

        job = _RLAJob(recs, truth, self.metrics)
        job.prepare(self.group_cols)

        timer = Stopwatch()
        _log.info('collecting metric results')
        res = job.compute(self.n_jobs)
        _log.info('measured %d lists in %s', len(res), timer)

        if include_missing:
            _log.info('filling in missing user info')
            tcount = truth.groupby(job.truth_key)['item'].count()
            tcount.name = 'ntruth'
            res = res.join(tcount, how='outer', on=job.truth_key)
            _log.debug('final columns: %s', res.columns)
            _log.debug('index levels: %s', res.index.names)
            res['ntruth'] = res['ntruth'].fillna(0)
            res['nrecs'] = res['nrecs'].fillna(0)

        return res


def _df_keys(r_cols, t_cols, g_cols=None, skip_cols=RecListAnalysis.DEFAULT_SKIP_COLS):
    "Identify rec list and truth list keys."
    if g_cols is None:
        g_cols = [c for c in r_cols if c not in skip_cols]

    truth_key = [c for c in g_cols if c in t_cols]
    rec_key = [c for c in g_cols if c not in t_cols] + truth_key
    _log.info('using rec key columns %s', rec_key)
    _log.info('using truth key columns %s', truth_key)
    return rec_key, truth_key
