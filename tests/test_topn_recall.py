import logging
import numpy as np
import pandas as pd

from pytest import approx

from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.basic import PopScore
from lenskit.algorithms.ranking import PlackettLuce
from lenskit.algorithms import Recommender
from lenskit.util.test import ml_test
from lenskit.metrics.topn import recall
from lenskit import topn, batch, crossfold as xf

_log = logging.getLogger(__name__)


def _test_recall(items, rel, **kwargs):
    recs = pd.DataFrame({'item': items})
    truth = pd.DataFrame({'item': rel}).set_index('item')
    return recall(recs, truth, **kwargs)


def test_recall_empty_zero():
    prec = _test_recall([], [1, 3])
    assert prec == approx(0)


def test_recall_norel_na():
    prec = _test_recall([1, 3], [])
    assert prec is None


def test_recall_simple_cases():
    prec = _test_recall([1, 3], [1, 3])
    assert prec == approx(1.0)

    prec = _test_recall([1], [1, 3])
    assert prec == approx(0.5)

    prec = _test_recall([1, 2, 3, 4], [1, 3])
    assert prec == approx(1.0)

    prec = _test_recall([1, 2, 3, 4], [1, 3, 5])
    assert prec == approx(2.0 / 3)

    prec = _test_recall([1, 2, 3, 4], range(5, 10))
    assert prec == approx(0.0)

    prec = _test_recall([1, 2, 3, 4], range(4, 9))
    assert prec == approx(0.2)


def test_recall_series():
    prec = _test_recall(pd.Series([1, 3]), pd.Series([1, 3]))
    assert prec == approx(1.0)

    prec = _test_recall(pd.Series([1, 2, 3]), pd.Series([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), pd.Series(range(4, 9)))
    assert prec == approx(0.2)


def test_recall_series_set():
    prec = _test_recall(pd.Series([1, 2, 3, 4]), [1, 3, 5, 7])
    assert prec == approx(0.5)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), range(4, 9))
    assert prec == approx(0.2)


def test_recall_series_index():
    prec = _test_recall(pd.Series([1, 3]), pd.Index([1, 3]))
    assert prec == approx(1.0)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), pd.Index([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), pd.Index(range(4, 9)))
    assert prec == approx(0.2)


def test_recall_series_array():
    prec = _test_recall(pd.Series([1, 3]), np.array([1, 3]))
    assert prec == approx(1.0)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), np.array([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), np.arange(4, 9, 1, 'u4'))
    assert prec == approx(0.2)


def test_recall_array():
    prec = _test_recall(np.array([1, 3]), np.array([1, 3]))
    assert prec == approx(1.0)

    prec = _test_recall(np.array([1, 2, 3, 4]), np.array([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = _test_recall(np.array([1, 2, 3, 4]), np.arange(4, 9, 1, 'u4'))
    assert prec == approx(0.2)


def test_recall_long_rel():
    rel = np.arange(100)
    items = [1, 0, 150, 3, 10]

    r = _test_recall(items, rel, k=5)
    assert r == approx(0.8)


def test_recall_long_truth():
    rel = np.arange(100)
    items = [1, 0, 150, 3, 10, 30, 120, 4, 17]

    r = _test_recall(items, rel, k=5)
    assert r == approx(0.8)


def test_recall_partial_rel():
    rel = np.arange(100)
    items = [1, 0, 150, 3, 10]

    r = _test_recall(items, rel, k=10)
    assert r == approx(0.4)


def test_recall_bulk_k():
    "bulk and normal match"
    train, test = xf.simple_test_pair(ml_test.ratings, f_rates=0.5)
    assert test['user'].value_counts().max() > 5

    users = test['user'].unique()
    algo = PopScore()
    algo = PlackettLuce(algo, rng_spec='user')
    algo.fit(train)

    recs = batch.recommend(algo, users, 1000)

    rla = topn.RecListAnalysis()
    rla.add_metric(recall, name='rk', k=5)
    rla.add_metric(recall)
    # metric without the bulk capabilities
    rla.add_metric(lambda *a, **k: recall(*a, **k), name='ind_rk', k=5)
    rla.add_metric(lambda *a: recall(*a), name='ind_r')
    res = rla.compute(recs, test)

    print(res)
    _log.info('recall mismatches:\n%s',
              res[res.recall != res.ind_r])

    assert res.recall.values == approx(res.ind_r.values)
    assert res.rk.values == approx(res.ind_rk.values)
