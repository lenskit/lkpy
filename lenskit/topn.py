import logging
import functools as ft

import numpy as np
import pandas as pd

from .metrics.topn import *
from .util import Stopwatch

_log = logging.getLogger(__name__)


def _length(df, *args, **kwargs):
    return float(len(df))


@bulk_impl(_length)
def _bulk_length(df, *args):
    return df.groupby('LKRecID')['item'].count()


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

        rec_key, truth_key = _df_keys(recs.columns, truth.columns, self.group_cols)

        t_ident, t_data = self._number_truth(truth, truth_key)
        r_ident, r_data = self._number_recs(recs, truth_key, rec_key, t_ident)

        timer = Stopwatch()

        _log.info('collecting metric results')
        bulk_res = []
        ind_metrics = []
        for mf, mn, margs in self.metrics:
            if hasattr(mf, 'bulk_score') and 'rank' in r_data.columns:
                _log.debug('bulk-scoring %s', mn)
                mbs = mf.bulk_score(r_data, t_data, **margs).to_frame(name=mn)
                assert mbs.index.name == 'LKRecID'
                bulk_res.append(mbs)
            else:
                ind_metrics.append((mf, mn, margs))
        if bulk_res:
            bulk_res = ft.reduce(lambda df1, df2: df1.join(df2, how='outer'), bulk_res)
        else:
            bulk_res = None

        def worker(rdf):
            rk, tk = rdf.name
            tdf = t_data.loc[tk]
            res = pd.Series(dict((mn, mf(rdf, tdf, **margs)) for (mf, mn, margs) in ind_metrics))
            return res

        if ind_metrics:
            _log.debug('applying individual metrics')
            groups = r_data.groupby(['LKRecID', 'LKTruthID'], sort=False)
            if hasattr(groups, 'progress_apply'):
                ind_res = groups.progress_apply(worker)
            else:
                ind_res = groups.apply(worker)
            ind_res = ind_res.reset_index('LKTruthID', drop=True)

            if bulk_res is not None:
                res = bulk_res.join(ind_res)
            else:
                res = ind_res
        else:
            res = bulk_res

        _log.debug('transforming results')
        res = r_ident.join(res, on='LKRecID').drop(columns=['LKRecID', 'LKTruthID'])

        _log.info('measured %d lists in %s', len(res), timer)

        if include_missing:
            _log.info('filling in missing user info (%d initial rows)', len(res))
            ug_cols = [c for c in rec_key if c not in truth_key]
            tcount = truth.groupby(truth_key)['item'].count().to_frame('ntruth')
            _log.debug('truth counts:\n%s', tcount)
            if ug_cols:
                _log.debug('regrouping by %s to fill', ug_cols)
                _log.debug('pre-group series:\n%s', res)

                rdict = {}

                for key, df in res.groupby(ug_cols):
                    df2 = df.drop(columns=ug_cols).join(tcount, how='outer', on=truth_key)
                    rdict[key] = df2

                res = pd.concat(rdict, names=ug_cols)
                _log.debug('joined result:\n%s', res)
                res = res.reset_index(ug_cols)
                res.reset_index(inplace=True, drop=True)
                _log.debug('final joined result:\n%s', res)

            else:
                _log.debug('no ungroup cols, directly merging to fill')
                res = res.join(tcount, how='outer', on=truth_key)
            _log.debug('final columns: %s', res.columns)
            _log.debug('index levels: %s', res.index.names)
            _log.debug('expanded to %d rows', len(res))
            res['ntruth'] = res['ntruth'].fillna(0)
            res['nrecs'] = res['nrecs'].fillna(0)

        return res.set_index(rec_key)

    def _number_truth(self, truth, truth_key):
        _log.info('numbering truth lists')
        truth_df = truth[truth_key].drop_duplicates()
        truth_df['LKTruthID'] = np.arange(len(truth_df))
        truth = pd.merge(truth_df, truth, on=truth_key).drop(columns=truth_key)

        truth.set_index(['LKTruthID', 'item'], inplace=True)
        if not truth.index.is_unique:
            _log.warn('truth index not unique: may have duplicate items\n%s', truth)

        return truth_df, truth

    def _number_recs(self, recs, truth_key, rec_key, t_ident):
        _log.info('numbering rec lists')
        rec_df = recs[rec_key].drop_duplicates()
        rec_df['LKRecID'] = np.arange(len(rec_df))
        rec_df = pd.merge(rec_df, t_ident, on=truth_key, how='left')
        recs = pd.merge(rec_df, recs, on=rec_key).drop(columns=rec_key)

        return rec_df, recs


def _df_keys(r_cols, t_cols, g_cols=None, skip_cols=RecListAnalysis.DEFAULT_SKIP_COLS):
    "Identify rec list and truth list keys."
    if g_cols is None:
        g_cols = [c for c in r_cols if c not in skip_cols]

    truth_key = [c for c in g_cols if c in t_cols]
    rec_key = [c for c in g_cols if c not in t_cols] + truth_key
    _log.info('using rec key columns %s', rec_key)
    _log.info('using truth key columns %s', truth_key)
    return rec_key, truth_key
