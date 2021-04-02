import numpy as np
import pandas as pd

from pytest import approx, mark

from lenskit.topn import RecListAnalysis, recip_rank
from lenskit.util.test import ml_test, demo_recs
from lenskit.algorithms.basic import Popular
from lenskit.batch import recommend
from lenskit.crossfold import simple_test_pair


def _test_rr(items, rel):
    recs = pd.DataFrame({'item': items})
    truth = pd.DataFrame({'item': rel}).set_index('item')
    return recip_rank(recs, truth)


def test_mrr_empty_zero():
    rr = _test_rr([], [1, 3])
    assert rr == approx(0)


def test_mrr_norel_zero():
    "no relevant items -> zero"
    rr = _test_rr([1, 2, 3], [4, 5])
    assert rr == approx(0)


def test_mrr_first_one():
    "first relevant -> one"
    rr = _test_rr([1, 2, 3], [1, 4])
    assert rr == approx(1.0)


def test_mrr_second_one_half():
    "second relevant -> 0.5"
    rr = _test_rr([1, 2, 3], [5, 2, 3])
    assert rr == approx(0.5)


def test_mrr_series():
    "second relevant -> 0.5 in pd series"
    rr = _test_rr(pd.Series([1, 2, 3]), pd.Series([5, 2, 3]))
    assert rr == approx(0.5)


def test_mrr_series_idx():
    "second relevant -> 0.5 in pd series w/ index"
    rr = _test_rr(pd.Series([1, 2, 3]), pd.Index([5, 2, 3]))
    assert rr == approx(0.5)


def test_mrr_array_late():
    "deep -> 0.1"
    rr = _test_rr(np.arange(1, 21, 1, 'u4'), [20, 10])
    assert rr == approx(0.1)


@mark.parametrize('drop_rating', [False, True])
def test_mrr_bulk(demo_recs, drop_rating):
    "bulk and normal match"
    train, test, recs = demo_recs
    if drop_rating:
        test = test[['user', 'item']]

    rla = RecListAnalysis()
    rla.add_metric(recip_rank)
    # metric without the bulk capabilities
    rla.add_metric(lambda *a: recip_rank(*a), name='ind_rr')
    res = rla.compute(recs, test)

    assert all(res.recip_rank == res.ind_rr)
