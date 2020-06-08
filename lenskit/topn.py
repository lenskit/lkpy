import logging
import itertools as it
import warnings
from collections import OrderedDict as od

import numpy as np
import pandas as pd

from .metrics.topn import *
from .util import Stopwatch
from .util.parallel import invoker

_log = logging.getLogger(__name__)


def _length(df, *args, **kwargs):
    return float(len(df))


def _grouping_iter(df, cols, ksf=()):
    # what key are we going to work on?
    current = cols[0]
    remaining = cols[1:]

    if not remaining:
        # this is the last column. we group by it.
        for gk, gdf in df.groupby(current):
            yield ksf + (gk,), gdf.drop(columns=[current])
    else:
        # we have columns remaining, let's start grouping by this one
        for v, subdf in df.groupby(current):
            yield from _grouping_iter(subdf.drop(columns=[current]), remaining, ksf + (v,))


def _rla_worker(model, req):
    col, val = req
    return model._compute_group(col, val)


class _RLAJob:
    def __init__(self, recs, truth, metrics):
        self.recs = recs
        self.truth = truth
        self.metrics = metrics

    def prepare(self, group_cols):
        rec_key, truth_key = _df_keys(self.recs.columns, self.truth.columns, group_cols)
        _log.info('collecting truth data')
        truth_frames = dict((k, df.set_index('item'))
                            for (k, df)
                            in _grouping_iter(self.truth, truth_key))
        _log.debug('found truth for %d users', len(truth_frames))
        self.truth = truth_frames
        self.rec_key = rec_key
        self.truth_key = truth_key

    def compute(self, n_jobs=None):
        first = self.rec_key[0]
        bins = self.recs.groupby(first)['item'].count()
        total = bins.sum()
        _log.debug('info RLA for %d rows in %d bins', total, len(bins))
        if total < 1000 or len(bins) < 4:
            n_jobs = 1  # force in-process for small runs

        with invoker(self, _rla_worker, n_jobs) as loop:
            res = loop.map((first, v) for v in bins.index.values)
            res = pd.concat(res, ignore_index=True)

        return res.set_index(self.rec_key)

    def _compute_group(self, col, val):
        _log.debug('computing for %s=%s', col, val)
        mnames = [mn for (mf, mn, margs) in self.metrics]
        gen = self._iter_measurements(col, val)
        return pd.DataFrame.from_records(gen, columns=self.rec_key + mnames)

    def _iter_measurements(self, col, val):
        key = self.rec_key[1:]
        df = self.recs[self.recs[col] == val]
        df = df.drop(columns=[col])
        nt = len(self.truth_key)
        if key:
            for rk, gdf in _grouping_iter(df, key):
                rk = (val,) + rk
                tk = rk[-nt:]
                g_truth = self.truth[tk]
                results = tuple(mf(gdf, g_truth, **margs) for (mf, mn, margs) in self.metrics)
                yield rk + results
        else:
            # we only have one group level
            tk = (val,)
            g_truth = self.truth[tk]
            results = tuple(mf(df, g_truth, **margs) for (mf, mn, margs) in self.metrics)
            yield tk + results


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
            ug_cols = [c for c in job.rec_key if c not in job.truth_key]
            tcount = truth.reset_index().groupby(job.truth_key)['item'].count()
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


def _df_keys(r_cols, t_cols, g_cols=None, skip_cols=RecListAnalysis.DEFAULT_SKIP_COLS):
    "Identify rec list and truth list keys."
    if g_cols is None:
        g_cols = [c for c in r_cols if c not in skip_cols]

    truth_key = [c for c in g_cols if c in t_cols]
    rec_key = [c for c in g_cols if c not in t_cols] + truth_key
    _log.info('using rec key columns %s', rec_key)
    _log.info('using truth key columns %s', truth_key)
    return rec_key, truth_key
