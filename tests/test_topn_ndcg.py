import numpy as np
import pandas as pd

from pytest import approx

import lenskit.metrics.topn as lm
import lk_test_utils as lktu


def test_dcg_empty():
    "empty should be zero"
    assert lm._dcg(np.array([])) == approx(0)


def test_dcg_zeros():
    assert lm._dcg(np.zeros(10)) == approx(0)


def test_dcg_single():
    "a single element should be scored at the right place"
    assert lm._dcg(np.array([0.5])) == approx(0.5)
    assert lm._dcg(np.array([0, 0.5])) == approx(0.5)
    assert lm._dcg(np.array([0, 0, 0.5])) == approx(0.5 / np.log2(3))
    assert lm._dcg(np.array([0, 0, 0.5, 0])) == approx(0.5 / np.log2(3))


def test_dcg_mult():
    "multiple elements should score correctly"
    assert lm._dcg(np.array([np.e, np.pi])) == approx(np.e + np.pi)
    assert lm._dcg(np.array([np.e, 0, 0, np.pi])) == approx(np.e + np.pi / np.log2(4))


def test_dcg_empty2():
    "empty should be zero"
    assert lm.dcg(np.array([])) == approx(0)


def test_dcg_zeros2():
    assert lm.dcg(np.zeros(10)) == approx(0)


def test_dcg_single2():
    "a single element should be scored at the right place"
    assert lm.dcg(np.array([0.5])) == approx(0.5)
    assert lm.dcg(np.array([0, 0.5])) == approx(0.5)
    assert lm.dcg(np.array([0, 0, 0.5])) == approx(0.5 / np.log2(3))
    assert lm.dcg(np.array([0, 0, 0.5, 0])) == approx(0.5 / np.log2(3))


def test_dcg_mult2():
    "multiple elements should score correctly"
    assert lm.dcg(np.array([np.e, np.pi])) == approx(np.e + np.pi)
    assert lm.dcg(np.array([np.e, 0, 0, np.pi])) == \
        approx((np.e + np.pi / np.log2(4)))


def test_ideal_dcg():
    ratings = lktu.ml_pandas.renamed.ratings
    ratings = ratings.loc[:, ['user', 'item', 'rating']]

    ml_dcg = lm.compute_ideal_dcgs(ratings)
    assert len(ml_dcg) < len(ratings)
    assert 'ideal_dcg' in ml_dcg.columns
    assert all(ml_dcg.ideal_dcg > 1)
