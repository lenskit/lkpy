import numpy as np
import pandas as pd

from pytest import approx

import lenskit.metrics.topn as lm


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


def test_ndcg_empty():
    "empty should be zero"
    assert lm.ndcg(np.array([])) == approx(0)


def test_ndcg_zeros():
    assert lm.ndcg(np.zeros(10)) == approx(0)


def test_ndcg_single():
    "a single element should be scored at the right place"
    assert lm.ndcg(np.array([0.5])) == approx(1)
    assert lm.ndcg(np.array([0, 0.5])) == approx(1)
    assert lm.ndcg(np.array([0, 0, 0.5])) == approx(1 / np.log2(3))
    assert lm.ndcg(np.array([0, 0, 0.5, 0])) == approx(1 / np.log2(3))


def test_ndcg_mult():
    "multiple elements should score correctly"
    assert lm.ndcg(np.array([np.e, np.pi])) == approx(1)
    assert lm.ndcg(np.array([np.e, 0, 0, np.pi])) == \
        approx((np.e + np.pi / np.log2(4)) / (np.e + np.pi))


def test_ndcg_score_items():
    "looked-up items should score correctly"
    scores = pd.Series([1, 1, 1, 1], index=['a', 'b', 'c', 'd'])

    assert lm.ndcg(scores, []) == approx(0)
    assert lm.ndcg(scores, ['a', 'b']) == approx(1)
    assert lm.ndcg(scores, ['a', 'b', 'c', 'd']) == approx(1)
    assert lm.ndcg(scores, ['d', 'b', 'a', 'c']) == approx(1)
    assert lm.ndcg(scores, ['d', 'b', 'z']) == approx(2 / (2 + 1/np.log2(3)))
    assert lm.ndcg(scores, ['d', 'z', 'b']) == approx((1 + 1/np.log2(3)) / (2 + 1/np.log2(3)))
