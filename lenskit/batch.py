"""
Batch-run predictors and recommenders for evaluation.
"""

import logging
import multiprocessing
from functools import partial
from collections import namedtuple
import pickle

import pandas as pd
import numpy as np

from . import sharing

_logger = logging.getLogger(__package__)

WorkContext = namedtuple('WorkContext', ['repo', 'algo', 'mkey', 'model'])
__predict_context = None


def predict_pairs(predictor, pairs):
    """
    Generate predictions for user-item pairs.  The provided predictor should be a
    function of two arguments: the user ID and a list of item IDs. It should return
    a dictionary or a :py:class:`pandas.Series` mapping item IDs to predictions.

    Args:
        predictor(callable): a rating predictor function.
        pairs(pandas.DataFrame):
            a data frame of (``user``, ``item``) pairs to predict for. If this frame also
            contains a ``rating`` column, it will be included in the result.

    Returns:
        pandas.DataFrame:
            a frame with columns ``user``, ``item``, and ``prediction`` containing
            the prediction results. If ``pairs`` contains a `rating` column, this
            result will also contain a `rating` column.
    """

    ures = (predictor(user, udf.item).reset_index(name='prediction').assign(user=user)
            for (user, udf) in pairs.groupby('user'))
    res = pd.concat(ures).loc[:, ['user', 'item', 'prediction']]
    if 'rating' in pairs:
        return pairs.join(res.set_index(['user', 'item']), on=('user', 'item'))
    return res


def _persist_generic_model(repo, model):
    data = pickle.dumps(model)
    data = np.frombuffer(data, np.uint8)
    return repo.share(data)


def _load_generic_model(repo, key):
    data = repo.resolve(key)
    data = data.tobytes()
    return pickle.loads(data)


def _init_predict_worker(repo, algo, mkey):
    global __predict_context

    _logger.info('resolving model %s in worker', mkey)
    try:
        model = algo.resolve_model(mkey, repo)
    except BaseException as e:
        _logger.error('could not resolve model: %s', e)
        raise e
    __predict_context = WorkContext(repo, algo, mkey, model)


def _run_predict(user, items):
    ctx = __predict_context
    res = ctx.algo.predict(ctx.model, user, items)
    return res.reset_index(name='prediction').assign(user=user)


def predict(algo, model, pairs, processes=None, repo=None):
    """
    Generate predictions for user-item pairs.

    Args:
        algo(algorithms.Predictor): an algorithm.
        model(any): a model for the algorithm.
        pairs(pandas.DataFrame):
            a data frame of (``user``, ``item``) pairs to predict for. If this frame also
            contains a ``rating`` column, it will be included in the result.

    Returns:
        pandas.DataFrame:
            a frame with columns ``user``, ``item``, and ``prediction`` containing
            the prediction results. If ``pairs`` contains a `rating` column, this
            result will also contain a `rating` column.
    """

    if processes == 1:
        return predict_pairs(partial(algo.predict, model), pairs)

    close_repo = False
    if repo is None:
        repo = sharing.repo()
        _logger.info('started new repository %s', repo)
        close_repo = True
    try:
        key = algo.share_model(model, repo)
        _logger.info('shared model to %s', key)
        with repo.client() as client, \
                multiprocessing.Pool(processes, _init_predict_worker, (client, algo, key)) as pool:
            # make iterator that extracts group info
            ures = ((user, udf.item) for (user, udf) in pairs.groupby('user'))
            # and blast it across our process pool
            ures = pool.starmap(_run_predict, ures, chunksize=10)
            # decode responses
            res = pd.concat(ures).loc[:, ['user', 'item', 'prediction']]
            if 'rating' in pairs:
                return pairs.join(res.set_index(['user', 'item']), on=('user', 'item'))
            pool.close()
            pool.join()
            return res

    finally:
        if close_repo:
            _logger.info('closing repository %s', repo)
            repo.close()


_MPState = namedtuple('_MPState', ['train', 'test', 'algo'])


def _run_mpjob(job: _MPState) -> bytes:
    train = pd.read_msgpack(job.train)
    _logger.info('training %s on %d rows', job.algo, len(train))
    model = job.algo.train(train)
    test = pd.read_msgpack(job.test)
    _logger.info('generating predictions with %s for %d pairs', job.algo, len(test))
    results = predict_pairs(partial(job.algo.predict, model), test)
    return results.to_msgpack()


def _mp_stateify(sets, algo):
    for train, test in sets:
        train_bytes = train.to_msgpack()
        test_bytes = test.to_msgpack()
        yield _MPState(train_bytes, test_bytes, algo)


def _run_spjob(algo, train, test):
    _logger.info('training %s on %d rows', algo, len(train))
    model = algo.train(train)
    _logger.info('generating predictions with %s for %d pairs', algo, len(test))
    results = predict_pairs(partial(algo.predict, model), test)
    return results


def multi_predict(sets, algo, processes=None):
    _logger.info('launching multi-predict with %s processes', processes)
    if processes is None or processes > 1:
        with multiprocessing.Pool(processes) as p:
            results = [pd.read_msgpack(rbs) for rbs in p.map(_run_mpjob, _mp_stateify(sets, algo))]
    else:
        results = [_run_spjob(algo, train, test) for train, test in sets]

    _logger.info('finished %d predict jobs', len(results))

    return pd.concat(results)
