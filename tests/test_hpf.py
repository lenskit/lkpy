import logging

from lenskit.algorithms import hpf

import pandas as pd
import numpy as np

from pytest import mark

import lk_test_utils as lktu

try:
    import hpfrec
    have_hpfrec = True
except ImportError:
    have_hpfrec = False

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame({'item': [1, 1, 2, 3],
                          'user': [10, 12, 10, 13],
                          'rating': [4.0, 3.0, 5.0, 2.0]})


@mark.slow
@mark.skipif(not have_hpfrec, reason='hpfrec not installed')
def test_hpf_train_large():
    algo = hpf.HPF(20)
    ratings = lktu.ml_pandas.renamed.ratings
    ratings = ratings.assign(rating=ratings.rating + 0.5)
    model = algo.train(ratings)

    assert model is not None
    assert model.n_users == ratings.user.nunique()
    assert model.n_items == ratings.item.nunique()


@mark.slow
@mark.eval
@mark.skipif(not have_hpfrec, reason='hpfrec not installed')
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_hpf_batch_accuracy():
    import lenskit.crossfold as xf
    from lenskit import batch, topn
    import lenskit.metrics.topn as lm

    ratings = lktu.ml100k.load_ratings()

    algo = hpf.HPF(25)

    def eval(train, test):
        _log.info('running training')
        train['rating'] = train.rating.astype(np.float_)
        model = algo.train(train)
        users = test.user.unique()
        _log.info('testing %d users', len(users))
        candidates = topn.UnratedCandidates(train)
        recs = batch.recommend(algo, model, users, 100, candidates, test)
        return recs

    folds = xf.partition_users(ratings, 5, xf.SampleFrac(0.2))
    recs = pd.concat(eval(train, test) for (train, test) in folds)

    _log.info('analyzing recommendations')
    ndcg = recs.groupby('user').rating.apply(lm.ndcg)
    _log.info('ndcg for users is %.4f', ndcg.mean())
    assert ndcg.mean() > 0
