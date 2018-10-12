import logging

from lenskit import topn
from lenskit.algorithms import als

import pandas as pd
import numpy as np

import pytest
from pytest import approx, mark

import lk_test_utils as lktu

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame({'item': [1, 1, 2, 3],
                          'user': [10, 12, 10, 13],
                          'rating': [4.0, 3.0, 5.0, 2.0]})


def test_als_basic_build():
    algo = als.ImplicitMF(20, iterations=10)
    model = algo.train(simple_df)

    assert model is not None
    assert isinstance(model, als.MFModel)


def test_als_predict_basic():
    algo = als.ImplicitMF(20, iterations=10)
    model = algo.train(simple_df)

    assert model is not None

    preds = algo.predict(model, 10, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= 0
    assert preds.loc[3] <= 5


def test_als_predict_bad_item():
    algo = als.ImplicitMF(20, iterations=10)
    model = algo.train(simple_df)

    assert model is not None

    preds = algo.predict(model, 10, [4])
    assert len(preds) == 1
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_als_predict_bad_user():
    algo = als.ImplicitMF(20, iterations=10)
    model = algo.train(simple_df)

    assert model is not None

    preds = algo.predict(model, 50, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


@mark.slow
def test_als_train_large():
    algo = als.ImplicitMF(20, iterations=20)
    ratings = lktu.ml_pandas.renamed.ratings
    model = algo.train(ratings)

    assert model is not None
    # FIXME Write more test assertions


@mark.slow
def test_als_train_large_noratings():
    algo = als.ImplicitMF(20, iterations=20)
    ratings = lktu.ml_pandas.renamed.ratings
    ratings = ratings.loc[:, ['user', 'item']]
    model = algo.train(ratings)

    assert model is not None
    # FIXME Write more test assertions


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_als_implicit_batch_accuracy():
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.topn as lm

    ratings = lktu.ml100k.load_ratings()

    algo = als.ImplicitMF(25, iterations=20)

    def eval(train, test):
        _log.info('running training')
        train['rating'] = train.rating.astype(np.float_)
        model = algo.train(train)
        _log.info('testing %d users', test.user.nunique())
        candidates = topn.UnratedCandidates(train)
        recs = batch.recommend(algo, model, test.user, 100, candidates)
        # combine with test ratings for relevance data
        recs = pd.merge(recs, test, how='left', on=('user', 'item'))
        # fill in missing 0s
        recs.loc[recs.rating.isna(), 'rating'] = 0
        return recs

    folds = xf.partition_users(ratings, 5, xf.SampleFrac(0.2))
    recs = pd.concat(eval(train, test) for (train, test) in folds)

    _log.info('analyzing recommendations')
    ndcg = recs.groupby('user').rating.apply(lm.ndcg)
    _log.info('ndcg for users is %.4f', ndcg)
    assert ndcg.mean() > 0
