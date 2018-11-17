"""
Batch-run predictors and recommenders for evaluation.
"""

import logging
import pathlib
import collections
from time import perf_counter
from functools import partial
import warnings
from multiprocessing.pool import Pool

import pandas as pd
import numpy as np

from .algorithms import Predictor, Recommender
from . import topn, util, sharing

try:
    import fastparquet
except ImportError:
    fastparquet = None

_logger = logging.getLogger(__name__)


def predict(algo, pairs, model=None):
    """
    Generate predictions for user-item pairs.  The provided algorithm should be a
    :py:class:`algorithms.Predictor` or a function of two arguments: the user ID and
    a list of item IDs. It should return a dictionary or a :py:class:`pandas.Series`
    mapping item IDs to predictions.

    Args:
        predictor(callable or :py:class:algorithms.Predictor):
            a rating predictor function or algorithm.
        pairs(pandas.DataFrame):
            a data frame of (``user``, ``item``) pairs to predict for. If this frame also
            contains a ``rating`` column, it will be included in the result.
        model(any): a model for the algorithm.

    Returns:
        pandas.DataFrame:
            a frame with columns ``user``, ``item``, and ``prediction`` containing
            the prediction results. If ``pairs`` contains a `rating` column, this
            result will also contain a `rating` column.
    """

    if isinstance(algo, Predictor):
        pfun = partial(algo.predict, model)
    else:
        pfun = algo

    def run(user, udf):
        res = pfun(user, udf.item)
        return pd.DataFrame({'user': user, 'item': res.index, 'prediction': res.values})

    ures = (run(user, udf) for (user, udf) in pairs.groupby('user'))
    res = pd.concat(ures)
    if 'rating' in pairs:
        return pairs.join(res.set_index(['user', 'item']), on=('user', 'item'))
    return res


def _recommend_user(algo, model, user, n, candidates):
    _logger.debug('generating recommendations for %s', user)
    res = algo.recommend(model, user, n, candidates)
    iddf = pd.DataFrame({'user': user, 'rank': np.arange(1, len(res) + 1)})
    return pd.concat([iddf, res], axis='columns')


def _recommend_seq(algo, model, users, n, candidates):
    if isinstance(candidates, dict):
        candidates = candidates.get
    algo = Recommender.adapt(algo)
    results = [_recommend_user(algo, model, user, n, candidates(user))
               for user in users]
    return results


def _recommend_init(algo, mkey, ckey, ccls, n):
    global __rec_model, __rec_algo, __rec_candidates, __rec_size
    ctx = sharing.context()
    sharing._push_context(ctx)

    __rec_algo = Recommender.adapt(algo)
    __rec_model = sharing.resolve(mkey, algo)

    if ccls:
        __rec_candidates = ccls.share_resolve(ckey)
    else:
        __rec_candidates = ckey
    __rec_size = n


def _recommend_worker(user):
    candidates = __rec_candidates(user)
    return _recommend_user(__rec_algo, __rec_model, user, __rec_size, candidates)


def recommend(algo, model, users, n, candidates, ratings=None, nprocs=None):
    """
    Batch-recommend for multiple users.  The provided algorithm should be a
    :py:class:`algorithms.Recommender` or :py:class:`algorithms.Predictor` (which
    will be converted to a top-N recommender).

    Args:
        algo: the algorithm
        model: The algorithm model
        users(array-like): the users to recommend for
        n(int): the number of recommendations to generate (None for unlimited)
        candidates:
            the users' candidate sets. This can be a function, in which case it will
            be passed each user ID; it can also be a dictionary, in which case user
            IDs will be looked up in it.
        ratings(pandas.DataFrame):
            if not ``None``, a data frame of ratings to attach to recommendations when
            available.

    Returns:
        A frame with at least the columns ``user``, ``rank``, and ``item``; possibly also
        ``score``, and any other columns returned by the recommender.
    """

    if nprocs and nprocs > 1 and sharing.is_sharable(model, algo):
        shared = sharing.publish(model, algo)
        if isinstance(candidates, sharing.Shareable):
            cand_key = candidates.share_publish()
            cand_cls = candidates.__class__
        else:
            cand_key = candidates
            cand_cls = None

        args = [algo, shared, cand_key, cand_cls, n]
        with Pool(nprocs, _recommend_init, args) as pool:
            results = pool.map(_recommend_worker, users)
    else:
        results = _recommend_seq(algo, model, users, n, candidates)

    results = pd.concat(results, ignore_index=True)

    if ratings is not None:
        # combine with test ratings for relevance data
        results = pd.merge(results, ratings, how='left', on=('user', 'item'))
        # fill in missing 0s
        results.loc[results.rating.isna(), 'rating'] = 0

    return results


_AlgoRec = collections.namedtuple('_AlgoRec', [
    'algorithm',
    'parallel',
    'attributes'
])
_DSRec = collections.namedtuple('_DSRec', [
    'dataset',
    'candidates',
    'attributes'
])


class MultiEval:
    """
    A runner for carrying out multiple evaluations, such as parameter sweeps.

    Args:
        path(str or :py:class:`pathlib.Path`):
            the working directory for this evaluation.
            It will be created if it does not exist.
        nrecs(int):
            the number of recommendations to generate per user (None to disable top-N).
        candidates:
            the default candidate set generator.
    """

    def __init__(self, path, nrecs=100, candidates=topn.UnratedCandidates,
                 nprocs=None):
        self.workdir = pathlib.Path(path)
        self.n_recs = nrecs
        self.candidate_generator = candidates
        self.algorithms = []
        self.datasets = []
        self.nprocs = nprocs

    @property
    def run_csv(self):
        return self.workdir / 'runs.csv'

    @property
    def run_file(self):
        return self.workdir / 'runs.parquet'

    @property
    def preds_file(self):
        return self.workdir / 'predictions.parquet'

    @property
    def recs_file(self):
        return self.workdir / 'recommendations.parquet'

    def add_algorithms(self, algos, parallel=False, attrs=[], **kwargs):
        """
        Add one or more algorithms to the run.

        Args:
            algos(algorithm or list): the algorithm(s) to add.
            parallel(bool):
                if ``True``, allow this algorithm to be trained in parallel with others.
            attrs(list of str):
                a list of attributes to extract from the algorithm objects and include in
                the run descriptions.
            kwargs:
                additional attributes to include in the run descriptions.
        """

        if not isinstance(algos, collections.Iterable):
            algos = [algos]

        for algo in algos:
            aa = {'AlgoClass': algo.__class__.__name__, 'AlgoStr': str(algo)}
            aa.update(kwargs)
            for an in attrs:
                aa[an] = getattr(algo, an, None)

            self.algorithms.append(_AlgoRec(algo, parallel, aa))

    def add_datasets(self, data, name=None, candidates=None, **kwargs):
        """
        Add one or more datasets to the run.

        Args:
            data:
                the input data set(s) to run. Can be one of the followin:

                * A tuple of (train, test) data.
                * An iterable of (train, test) pairs, in which case the iterable
                  is not consumed until it is needed.
                * A function yielding either of the above, to defer data load
                  until it is needed.

            kwargs:
                additional attributes pertaining to these data sets.
        """

        attrs = {}
        if name is not None:
            attrs['DataSet'] = name
        attrs.update(kwargs)

        self.datasets.append(_DSRec(data, candidates, attrs))

    def run(self):
        """
        Run the evaluation.
        """

        self.workdir.mkdir(parents=True, exist_ok=True)

        run_id = 0
        run_data = []

        for ds, cand_f, ds_attrs in self.datasets:
            # normalize data set to be an iterable of tuples
            if callable(ds):
                ds = ds()
            if isinstance(ds, tuple):
                ds = [ds]
            if cand_f is None:
                cand_f = self.candidate_generator

            # loop the data
            for part, (train, test) in enumerate(ds):
                dsp_attrs = dict(ds_attrs)
                dsp_attrs['Partition'] = part + 1
                ds_name = ds_attrs.get('DataSet', None)
                cand = cand_f(train)

                for arec in self.algorithms:
                    run_id += 1

                    _logger.info('starting run %d: %s on %s:%d', run_id, arec.algorithm,
                                 ds_name, part + 1)
                    run = self._run_algo(run_id, arec, (train, test, dsp_attrs, cand))
                    _logger.info('finished run %d: %s on %s:%d', run_id, arec.algorithm,
                                 ds_name, part + 1)
                    run_data.append(run)
                    run_df = pd.DataFrame(run_data)
                    # overwrite files to show progress
                    run_df.to_csv(self.run_csv)
                    run_df.to_parquet(self.run_file, compression=None)

    def _run_algo(self, run_id, arec, data):
        train, test, dsp_attrs, cand = data

        run = {'RunId': run_id}
        run.update(dsp_attrs)
        run.update(arec.attributes)

        model, train_time = self._train_algo(arec.algorithm, train)
        run['TrainTime'] = train_time

        preds, pred_time = self._predict(run_id, arec.algorithm, model, test)
        run['PredTime'] = pred_time
        self._write_results(self.preds_file, preds, append=run_id > 1)

        recs, rec_time = self._recommend(run_id, arec.algorithm, model, test, cand)
        run['recTime'] = rec_time
        self._write_results(self.recs_file, recs, append=run_id > 1)

        return run

    def _train_algo(self, algo, train):
        watch = util.Stopwatch()
        _logger.info('training algorithm %s on %d ratings', algo, len(train))
        model = algo.train(train)
        watch.stop()
        _logger.info('trained algorithm %s in %s', algo, watch)
        return model, watch.elapsed()

    def _predict(self, rid, algo, model, test):
        if not isinstance(algo, Predictor):
            return None, None

        watch = util.Stopwatch()
        _logger.info('generating %d predictions for %s', len(test), algo)
        preds = predict(algo, test, model)
        watch.stop()
        _logger.info('generated predictions in %s', watch)
        preds['RunId'] = rid
        preds = preds[['RunId', 'user', 'item', 'rating', 'prediction']]
        return preds, watch.elapsed()

    def _recommend(self, rid, algo, model, test, candidates):
        if self.n_recs is None:
            return None, None

        watch = util.Stopwatch()
        users = test.user.unique()
        _logger.info('generating recommendations for %d users for %s', len(users), algo)
        recs = recommend(algo, model, users, self.n_recs, candidates, test,
                         nprocs=self.nprocs)
        watch.stop()
        _logger.info('generated recommendations in %s', watch)
        recs['RunId'] = rid
        return recs, watch.elapsed()

    def _write_results(self, file, df, append=True):
        if df is None:
            return

        if fastparquet is not None:
            fastparquet.write(str(file), df, append=append, compression='snappy')
        elif append and file.exists():
            warnings.warn('fastparquet not available, appending is slow')
            odf = pd.read_parquet(str(file))
            pd.concat([odf, df]).to_parquet(str(file))
        else:
            df.to_parquet(str(file))
