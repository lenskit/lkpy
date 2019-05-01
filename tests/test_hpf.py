import logging
import pickle

from lenskit.algorithms import hpf, basic

import pandas as pd
import numpy as np

from pytest import mark

import lenskit.util.test as lktu

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
def test_hpf_train_large(tmp_path):
    algo = hpf.HPF(20)
    ratings = lktu.ml_test.ratings
    ratings = ratings.assign(rating=ratings.rating + 0.5)
    algo.fit(ratings)

    assert algo.n_users == ratings.user.nunique()
    assert algo.n_items == ratings.item.nunique()

    mfile = tmp_path / 'hpf.dat'
    with mfile.open('wb') as mf:
        pickle.dump(algo, mf)

    with mfile.open('rb') as mf:
        a2 = pickle.load(mf)

    assert a2.n_users == algo.n_users
    assert a2.n_items == algo.n_items

    csel = basic.UnratedItemCandidateSelector()
    csel.fit(ratings)
    rec = basic.TopN(algo, csel)

    for u in np.random.choice(ratings.user.unique(), size=50, replace=False):
        recs = rec.recommend(u, 50)
        assert len(recs) == 50
        assert recs.item.nunique() == 50


@mark.slow
@mark.skipif(not have_hpfrec, reason='hpfrec not installed')
def test_hpf_train_binary(tmp_path):
    algo = hpf.HPF(20)
    ratings = lktu.ml_test.ratings.drop(columns=['timestamp', 'rating'])
    algo.fit(ratings)

    assert algo.n_users == ratings.user.nunique()
    assert algo.n_items == ratings.item.nunique()

    mfile = tmp_path / 'hpf.dat'
    with mfile.open('wb') as mf:
        pickle.dump(algo, mf)

    with mfile.open('rb') as mf:
        a2 = pickle.load(mf)

    assert a2.n_users == algo.n_users
    assert a2.n_items == algo.n_items

    csel = basic.UnratedItemCandidateSelector()
    csel.fit(ratings)
    rec = basic.TopN(algo, csel)

    for u in np.random.choice(ratings.user.unique(), size=50, replace=False):
        recs = rec.recommend(u, 50)
        assert len(recs) == 50
        assert recs.item.nunique() == 50
