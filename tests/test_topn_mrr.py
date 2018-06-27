import numpy as np
import pandas as pd

from pytest import approx

import lenskit.metrics.topn as lm


def test_mrr_empty_zero():
    rr = lm.recip_rank([], [1, 3])
    assert rr == approx(0)


def test_mrr_norel_zero():
    "no relevant items -> zero"
    rr = lm.recip_rank([1, 2, 3], [4, 5])
    assert rr == approx(0)


def test_mrr_first_one():
    "first relevant -> one"
    rr = lm.recip_rank([1, 2, 3], [1, 4])
    assert rr == approx(1.0)


def test_mrr_second_one_half():
    "second relevant -> 0.5"
    rr = lm.recip_rank([1, 2, 3], [5, 2, 3])
    assert rr == approx(0.5)


def test_mrr_series():
    "second relevant -> 0.5 in pd series"
    rr = lm.recip_rank(pd.Series([1, 2, 3]), pd.Series([5, 2, 3]))
    assert rr == approx(0.5)


def test_mrr_series_idx():
    "second relevant -> 0.5 in pd series w/ index"
    rr = lm.recip_rank(pd.Series([1, 2, 3]), pd.Index([5, 2, 3]))
    assert rr == approx(0.5)


def test_mrr_array_late():
    "deep -> 0.1"
    rr = lm.recip_rank(np.arange(1, 21, 1, 'u4'), [20, 10])
    assert rr == approx(0.1)
