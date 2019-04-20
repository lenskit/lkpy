import numpy as np
import pandas as pd

from pytest import approx

from lenskit.metrics.topn import _dcg, ndcg
import lenskit.util.test as lktu


def test_dcg_empty():
    "empty should be zero"
    assert _dcg(np.array([])) == approx(0)


def test_dcg_zeros():
    assert _dcg(np.zeros(10)) == approx(0)


def test_dcg_single():
    "a single element should be scored at the right place"
    assert _dcg(np.array([0.5])) == approx(0.5)
    assert _dcg(np.array([0, 0.5])) == approx(0.5)
    assert _dcg(np.array([0, 0, 0.5])) == approx(0.5 / np.log2(3))
    assert _dcg(np.array([0, 0, 0.5, 0])) == approx(0.5 / np.log2(3))


def test_dcg_mult():
    "multiple elements should score correctly"
    assert _dcg(np.array([np.e, np.pi])) == approx(np.e + np.pi)
    assert _dcg(np.array([np.e, 0, 0, np.pi])) == approx(np.e + np.pi / np.log2(4))


def test_dcg_empty2():
    "empty should be zero"
    assert _dcg(np.array([])) == approx(0)


def test_dcg_zeros2():
    assert _dcg(np.zeros(10)) == approx(0)


def test_dcg_single2():
    "a single element should be scored at the right place"
    assert _dcg(np.array([0.5])) == approx(0.5)
    assert _dcg(np.array([0, 0.5])) == approx(0.5)
    assert _dcg(np.array([0, 0, 0.5])) == approx(0.5 / np.log2(3))
    assert _dcg(np.array([0, 0, 0.5, 0])) == approx(0.5 / np.log2(3))


def test_dcg_nan():
    "NANs should be 0"
    assert _dcg(np.array([np.nan, 0.5])) == approx(0.5)


def test_dcg_series():
    "The DCG function should work on a series"
    assert _dcg(pd.Series([np.e, 0, 0, np.pi])) == \
        approx((np.e + np.pi / np.log2(4)))


def test_dcg_mult2():
    "multiple elements should score correctly"
    assert _dcg(np.array([np.e, np.pi])) == approx(np.e + np.pi)
    assert _dcg(np.array([np.e, 0, 0, np.pi])) == \
        approx((np.e + np.pi / np.log2(4)))


def test_ndcg_empty():
    recs = pd.DataFrame({'item': []})
    truth = pd.DataFrame({'item': [1, 2, 3], 'rating': [3.0, 5.0, 4.0]})
    truth = truth.set_index('item')
    assert ndcg(recs, truth) == approx(0.0)


def test_ndcg_no_match():
    recs = pd.DataFrame({'item': [4]})
    truth = pd.DataFrame({'item': [1, 2, 3], 'rating': [3.0, 5.0, 4.0]})
    truth = truth.set_index('item')
    assert ndcg(recs, truth) == approx(0.0)


def test_ndcg_perfect():
    recs = pd.DataFrame({'item': [2, 3, 1]})
    truth = pd.DataFrame({'item': [1, 2, 3], 'rating': [3.0, 5.0, 4.0]})
    truth = truth.set_index('item')
    assert ndcg(recs, truth) == approx(1.0)


def test_ndcg_wrong():
    recs = pd.DataFrame({'item': [1, 2]})
    truth = pd.DataFrame({'item': [1, 2, 3], 'rating': [3.0, 5.0, 4.0]})
    truth = truth.set_index('item')
    assert ndcg(recs, truth) == approx(_dcg([3.0, 5.0] / _dcg([5.0, 4.0, 3.0])))
